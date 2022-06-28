from typing import List, Set, Tuple, Dict, Union

import torch
import numpy as np
from collections import defaultdict
import time
import sys
import os
import string
import json
import random

import spacy
from spacy.lang.en import STOP_WORDS
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY

from kb.wiki_linking_util import WikiCandidateMentionGenerator, MentionGenerator
from kb.entity_linking import TokenCharactersIndexerTokenizer
from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from kb.common import WhitespaceTokenizer, MentionGenerator, get_empty_candidates



import logging
import codecs

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator, start_token, sep_token

from itertools import chain








from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
from kb.bert_pretraining_reader import BertTokenizerCandidateGeneratorMasker
from kb.common import get_dtype_for_module, set_requires_grad, \
    extend_attention_mask_for_bert, init_bert_weights, F1Metric
from kb.dict_field import DictField




import copy


from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

from kb.dict_field import DictField

from kb.common import MentionGenerator, get_empty_candidates

start_token = "[CLS]"
sep_token = "[SEP]"




@MentionGenerator.register("wiki")
class WikiCandidateMentionGenerator(MentionGenerator):

    defaults = {
        "candidates_file": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/prob_yago_crosswikis_wikipedia_p_e_m.txt",
        "wiki_file": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/wiki_name_id_map.txt",
        "redirections_file": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/wiki_redirects.txt",
        "disambiguations_file": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/wiki_disambiguation_pages.txt",
        "entity_world_path": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/wiki_id_to_string.json",
    }

    def __init__(self,
                 candidates_file: str = None,
                 entity_world_path: str = None,
                 lowercase_candidates: bool = True,
                 random_candidates: bool = False,
                 pickle_cache_file: str = None,
                 ):

        self.tokenizer = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
        self.whitespace_tokenizer = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
        self.whitespace_tokenizer.tokenizer = WhitespaceTokenizer(self.whitespace_tokenizer.vocab)

        self.random_candidates = random_candidates
        self.lowercase_candidates = lowercase_candidates

        if isinstance(entity_world_path, dict):
            self.entity_world = entity_world_path
        else:
            entity_world_path = cached_path(entity_world_path or self.defaults["entity_world_path"])
            self.entity_world = json.load(open(entity_world_path))

        if pickle_cache_file is not None:
            import pickle
            with open(cached_path(pickle_cache_file), 'rb') as fin:
                data = pickle.load(fin)
            self.p_e_m = data['p_e_m']
            self.p_e_m_low = data['p_e_m_low']
            self.mention_total_freq = data['mention_total_freq']
        else:
            valid_candidates_with_vectors = set(self.entity_world.keys())
            candidates_file = cached_path(candidates_file or self.defaults["candidates_file"])
            self.p_e_m, self.p_e_m_low, self.mention_total_freq = prior_entity_candidates(candidates_file,
                                                                                      allowed_entities_set=valid_candidates_with_vectors)

        self.random_candidates = random_candidates
        if self.random_candidates:
            self.p_e_m_keys_for_sampling = list(self.p_e_m.keys())

    def get_mentions_raw_text(self, text: str, whitespace_tokenize=False):
        """
        returns:
            {'tokenized_text': List[str],
             'candidate_spans': List[List[int]] list of (start, end) indices for candidates,
                    where span is tokenized_text[start:(end + 1)]
             'candidate_entities': List[List[str]] = for each entity,
                    the candidates to link to. value is synset id, e.g
                    able.a.02 or hot_dog.n.01
             'candidate_entity_priors': List[List[float]]
        }
        """
        if whitespace_tokenize:
            tokens = self.whitespace_tokenizer(text)
        else:
            tokens = self.tokenizer(text)

        tokens = [t.text for t in tokens]
        all_spans = enumerate_spans(tokens, max_span_width=5, filter_function=span_filter_func)

        spans_to_candidates = {}

        for span in all_spans:
            candidate_entities = self.process(tokens[span[0]:span[1] + 1])
            if candidate_entities:
                # Only keep spans which we have candidates for.
                spans_to_candidates[(span[0], span[1])] = candidate_entities


        spans = []
        entities = []
        priors = []
        for span, candidates in spans_to_candidates.items():
            spans.append(list(span))
            entities.append([x[1] for x in candidates])
            mention_priors = [x[2] for x in candidates]

            # priors may not be normalized because we merged the
            # lowercase + cased values.
            sum_priors = sum(mention_priors)
            priors.append([x/sum_priors for x in mention_priors])

        ret = {
            "tokenized_text": tokens,
            "candidate_spans": spans,
            "candidate_entities": entities,
            "candidate_entity_priors": priors
        }

        if len(spans) == 0:
            ret.update(get_empty_candidates())

        return ret


    def get_mentions_with_gold(self, text: str, gold_spans, gold_entities, whitespace_tokenize=True, keep_gold_only: bool = False):

        gold_spans_to_entities = {tuple(k):v for k,v in zip(gold_spans, gold_entities)}

        if whitespace_tokenize:
            tokens = self.whitespace_tokenizer(text)
        else:
            tokens = self.tokenizer(text)

        tokens = [t.text for t in tokens]
        if keep_gold_only:
            spans_with_gold = set(gold_spans_to_entities.keys())
        else:
            all_spans = enumerate_spans(tokens, max_span_width=5, filter_function=span_filter_func)
            spans_with_gold = set().union(all_spans, [tuple(span) for span in gold_spans])

        spans = []
        entities = []
        gold_entities = []
        priors = []
        for span in spans_with_gold:
            candidate_entities = self.process(tokens[span[0]:span[1] + 1])

            gold_entity = gold_spans_to_entities.get(span, "@@NULL@@")
            # Only keep spans which we have candidates for.
            # For a small number of gold candidates,
            # we don't have mention candidates for them,
            # we can't link to them.
            if not candidate_entities:
                continue

            candidate_names = [x[1] for x in candidate_entities]
            candidate_priors = [x[2] for x in candidate_entities]
            sum_priors = sum(candidate_priors)
            priors.append([x/sum_priors for x in candidate_priors])

            spans.append(list(span))
            entities.append(candidate_names)
            gold_entities.append(gold_entity)

        return {
            "tokenized_text": tokens,
            "candidate_spans": spans,
            "candidate_entities": entities,
            # TODO Change to priors
            "candidate_entity_prior": priors,
            "gold_entities": gold_entities
        }


    def process(self, span: Union[List[str], str], lower=False) -> List[Tuple[str, str, float]]:
        """
        Look up spans in the candidate dictionary, including looking for
        a title format version of the same string. Returns a list of
        (entity_id, entity_candidate, p(entity_candidate | mention string)) pairs.
        """
        if self.random_candidates:
            random_key = random.choice(self.p_e_m_keys_for_sampling)
            return self.p_e_m[random_key]

        if isinstance(span, list):
            span = ' '.join(span)

        # Makes all first chars of words uppercase, eg barack obama -> Barack Obama.
        title = span.title()
        title_freq = self.mention_total_freq.get(title, 0)
        span_freq = self.mention_total_freq.get(span, 0)

        if title_freq == 0 and span_freq == 0:
            if lower and span.lower() in self.p_e_m:
                return self.p_e_m[span.lower()]
            elif self.lowercase_candidates and span.lower() in self.p_e_m_low:
                return self.p_e_m_low[span.lower()]
            else:
                return []
        else:
            if span_freq > title_freq:
                return self.p_e_m[span]
            else:
                return self.p_e_m[title]




@DatasetReader.register("aida_wiki_linking")
class LinkingReader(DatasetReader):

    """
    Reads entity linking data with the following format:

    boycott
    MMSTART_31717 TAB United_Kingdom
    British
    MMEND
    lamb
    .
    *NL*

    I.e one word per line, with `MMSTART_{wiki_id}` denoting the begining of entities and
    *NL* denoting new line boundaries.

    Documents are separated with:

    DOCSTART_4_CHINA
    ...
    DOCEND
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 entity_indexer: TokenIndexer = TokenIndexer.from_params(Params(INDEXER_DEFAULT)),
                 granularity: str = "sentence",
                 mention_generator: MentionGenerator = None,
                 should_remap_span_indices: bool = True,
                 entity_disambiguation_only: bool = False,
                 extra_candidate_generators: Dict[str, MentionGenerator] = None):

        lazy = False
        super().__init__(lazy)
        self.token_indexers = token_indexers or {"token": SingleIdTokenIndexer("token")}
        self.entity_indexer = {"ids": entity_indexer}
        self.separator = {"*NL*"}
        if granularity == "sentence":
            self.separator.add(".")

        if granularity not in {"sentence", "paragraph"}:
            raise ConfigurationError("Valid arguments for granularity are 'sentence' or 'paragraph'.")

        self.entity_disambiguation_only = entity_disambiguation_only
        self.mention_generator = mention_generator or WikiCandidateMentionGenerator()
        self.should_remap_span_indices = should_remap_span_indices

        self.extra_candidate_generators = extra_candidate_generators

    def _read(self, file_path: str):

        file_path = cached_path(file_path)
        words = []
        gold_spans = []
        gold_entities = []
        in_mention = False
        doc_id = None
        with open(file_path, encoding='utf-8') as input_file:
            for line in input_file:
                line = line.rstrip()
                if line in self.separator and not in_mention:
                    if line == ".":
                        words.append(line)
                    # if we have continuous *NL* *NL* do not return empty chunks
                    if len(words) > 0:
                        processed = self.mention_generator.get_mentions_with_gold(" ".join(words), gold_spans,
                                                                                  gold_entities, whitespace_tokenize=True, keep_gold_only=self.entity_disambiguation_only)
                        if processed["candidate_spans"]:
                            yield self.text_to_instance(doc_id=doc_id, **processed)
                    # Reset state
                    words = []
                    gold_spans = []
                    gold_entities = []

                elif line.startswith('MMSTART_'):
                    in_mention = True
                    _, name = line.split("\t")
                    name = name.strip()
                    gold_entities.append(name)
                    gold_spans.append([len(words)])
                elif line == 'MMEND':
                    in_mention = False
                    # Spans are inclusive in allennlp
                    gold_spans[-1].append(len(words) - 1)
                elif line.startswith('DOCSTART_'):
                    # ignore doc indicators
                    doc_id = line.strip()
                elif line.startswith('DOCEND'):
                    doc_id = None
                else:
                    words.append(line)
        if words:
            processed = self.mention_generator.get_mentions_with_gold(" ".join(words), gold_spans,
                                                                      gold_entities, whitespace_tokenize=True, keep_gold_only=self.entity_disambiguation_only)
            if processed["candidate_spans"]:
                yield self.text_to_instance(doc_id=doc_id, **processed)

    def text_to_instance(self,
                         tokenized_text: List[str],
                         candidate_entities: List[List[str]],
                         candidate_spans: List[List[int]],
                         candidate_entity_prior: List[List[float]],
                         gold_entities: List[str] = None,
                         doc_id: str = None):

        assert doc_id is not None

        token_field = TextField([Token(x) for x in tokenized_text], self.token_indexers)
        span_fields = ListField([SpanField(*span, token_field) for span in candidate_spans])

        candidate_entities = TextField(
                [Token(" ".join(candidate_list)) for candidate_list in candidate_entities],
                token_indexers=self.entity_indexer)

        max_cands = max(len(p) for p in candidate_entity_prior)
        for p in candidate_entity_prior:
            if len(p) < max_cands:
                p.extend([0.0] * (max_cands - len(p)))
        np_prior = np.array(candidate_entity_prior)
        prior_field = ArrayField(np_prior)

        # only one segment
        candidate_segment_ids = ArrayField(
                np.array([0] * len(candidate_entities)), dtype=np.int
        )

        fields = {
            "tokens": token_field,
            "candidate_spans": span_fields,
            "candidate_entities": candidate_entities,
            "candidate_entity_prior": prior_field,
            "candidate_segment_ids": candidate_segment_ids
            }
        if gold_entities:
            labels = TextField([Token(entity) for entity in gold_entities],
                               token_indexers=self.entity_indexer)
            fields["gold_entities"] = labels

        fields["doc_id"] = MetadataField(doc_id)

        if self.extra_candidate_generators:
            tokens = " ".join(tokenized_text)
            extra_candidates = {
                    key: generator.get_mentions_raw_text(tokens, whitespace_tokenize=True)
                    for key, generator in self.extra_candidate_generators.items()
            }
            fields['extra_candidates'] = MetadataField(extra_candidates)

        return Instance(fields, should_remap_span_indices=self.should_remap_span_indices)



# ====================================
class StubbedCandidateGenerator:
    # this is dangerous, we will stub out the candidate generator to do a look
    # up operation from the known candidates

    def set_cache(self, candidates):
        self.cached_candidates = candidates

    def get_mentions_raw_text(self, text, whitespace_tokenize=True):
        return copy.deepcopy(self.cached_candidates[text])



def truncate_sequence_pair(word_piece_tokens_a, word_piece_tokens_b, max_word_piece_sequence_length):
    length_a = sum([len(x) for x in word_piece_tokens_a])
    length_b = sum([len(x) for x in word_piece_tokens_b])
    while max_word_piece_sequence_length < length_a + length_b:
        if length_a < length_b:
            discarded = word_piece_tokens_b.pop()
            length_b -= len(discarded)
        else:
            discarded = word_piece_tokens_a.pop()
            length_a -= len(discarded)





@TokenizerAndCandidateGenerator.register("bert_tokenizer_and_candidate_generator")
class BertTokenizerAndCandidateGenerator(Registrable):
    def __init__(self,
                 entity_candidate_generators: Dict[str, MentionGenerator],
                 entity_indexers: Dict[str, TokenIndexer],
                 bert_model_type: str,
                 do_lower_case: bool,
                 whitespace_tokenize: bool = True,
                 max_word_piece_sequence_length: int = 512) -> None:
        """
        Note: the fields need to be used with a pre-generated allennlp vocabulary
        that contains the entity id namespaces and the bert name space.
        entity_indexers = {'wordnet': indexer for wordnet entities,
                          'wiki': indexer for wiki entities}
        """
        # load BertTokenizer from huggingface
        self.candidate_generators = entity_candidate_generators
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            bert_model_type, do_lower_case=do_lower_case
        )
        self.bert_word_tokenizer = BasicTokenizer(do_lower_case=False)
        # Target length should include start and end token
        self.max_word_piece_sequence_length = max_word_piece_sequence_length

        self._entity_indexers = entity_indexers
        # for bert, we'll give an empty token indexer with empty name space
        # and do the indexing directly with the bert vocab to bypass
        # indexing in the indexer
        self._bert_single_id_indexer = {'tokens': SingleIdTokenIndexer('__bert__')}
        self.do_lowercase = do_lower_case
        self.whitespace_tokenize = whitespace_tokenize
        self.dtype = np.float32

    def _word_to_word_pieces(self, word):
        if self.do_lowercase and word not in self.bert_tokenizer.basic_tokenizer.never_split:
            word = word.lower()
        return self.bert_tokenizer.wordpiece_tokenizer.tokenize(word)

    def tokenize_and_generate_candidates(self, text_a: str, text_b: str = None):
        """
        # run BertTokenizer.basic_tokenizer.tokenize on sentence1 and sentence2 to word tokenization
        # generate candidate mentions for each of the generators and for each of sentence1 and 2 from word tokenized text
        # run BertTokenizer.wordpiece_tokenizer on sentence1 and sentence2
        # truncate length, add [CLS] and [SEP] to word pieces
        # compute token offsets
        # combine candidate mention spans from sentence1 and sentence2 and remap to word piece indices

        returns:

        {'tokens': List[str], the word piece strings with [CLS] [SEP]
         'segment_ids': List[int] the same length as 'tokens' with 0/1 for sentence1 vs 2
         'candidates': Dict[str, Dict[str, Any]],
            {'wordnet': {'candidate_spans': List[List[int]],
                         'candidate_entities': List[List[str]],
                         'candidate_entity_prior': List[List[float]],
                         'segment_ids': List[int]},
             'wiki': ...}
        }
        """
        offsets_a, grouped_wp_a, tokens_a = self._tokenize_text(text_a)

        if text_b is not None:
            offsets_b, grouped_wp_b, tokens_b = self._tokenize_text(text_b)
            truncate_sequence_pair(grouped_wp_a, grouped_wp_b, self.max_word_piece_sequence_length - 3)
            offsets_b = offsets_b[:len(grouped_wp_b)]
            tokens_b = tokens_b[:len(grouped_wp_b)]
            instance_b = self._generate_sentence_entity_candidates(tokens_b, offsets_b)
            word_piece_tokens_b = [word_piece for word in grouped_wp_b for word_piece in word]
        else:
            length_a = sum([len(x) for x in grouped_wp_a])
            while self.max_word_piece_sequence_length - 2 < length_a:
                discarded = grouped_wp_a.pop()
                length_a -= len(discarded)

        word_piece_tokens_a = [word_piece for word in grouped_wp_a for word_piece in word]
        offsets_a = offsets_a[:len(grouped_wp_a)]
        tokens_a = tokens_a[:len(grouped_wp_a)]
        instance_a = self._generate_sentence_entity_candidates(tokens_a, offsets_a)

        # If we got 2 sentences.
        if text_b is not None:
            # Target length should include start and two end tokens, and then be divided equally between both sentences
            # Note that this will result in potentially shorter documents than original target length,
            # if one (or both) of the sentences are shorter than half the target length.
            tokens = [start_token] + word_piece_tokens_a + [sep_token] + word_piece_tokens_b + [sep_token]
            segment_ids = (len(word_piece_tokens_a) + 2) * [0] + (len(word_piece_tokens_b) + 1) * [1]
            offsets_a = [x + 1 for x in offsets_a]
            offsets_b = [x + 2 + len(word_piece_tokens_a) for x in offsets_b]
        # Single sentence
        else:
            tokens = [start_token] + word_piece_tokens_a + [sep_token]
            segment_ids = len(tokens) * [0]
            offsets_a = [x + 1 for x in offsets_a]
            offsets_b = None

        for name in instance_a.keys():
            for span in instance_a[name]['candidate_spans']:
                span[0] += 1
                span[1] += 1

        fields: Dict[str, Sequence] = {}

        # concatanating both sentences (for both tokens and ids)
        if text_b is None:
            candidates = instance_a
        else:
            candidates: Dict[str, Field] = {}

            # Merging candidate lists for both sentences.
            for entity_type in instance_b:
                candidate_instance_a = instance_a[entity_type]
                candidate_instance_b = instance_b[entity_type]

                candidates[entity_type] = {}

                for span in candidate_instance_b['candidate_spans']:
                    span[0] += len(word_piece_tokens_a) + 2
                    span[1] += len(word_piece_tokens_a) + 2

                # Merging each of the fields.
                for key in ['candidate_entities', 'candidate_spans', 'candidate_entity_priors']:
                    candidates[entity_type][key] = candidate_instance_a[key] + candidate_instance_b[key]


        for entity_type in candidates.keys():
            # deal with @@PADDING@@
            if len(candidates[entity_type]['candidate_entities']) == 0:
                candidates[entity_type] = get_empty_candidates()
            else:
                padding_indices = []
                has_entity = False
                for cand_i, candidate_list in enumerate(candidates[entity_type]['candidate_entities']):
                    if candidate_list == ["@@PADDING@@"]:
                        padding_indices.append(cand_i)
                        candidates[entity_type]["candidate_spans"][cand_i] = [-1, -1]
                    else:
                        has_entity = True
                indices_to_remove = []
                if has_entity and len(padding_indices) > 0:
                    # remove all the padding entities since have some valid
                    indices_to_remove = padding_indices
                elif len(padding_indices) > 0:
                    assert len(padding_indices) == len(candidates[entity_type]['candidate_entities'])
                    indices_to_remove = padding_indices[1:]
                for ind in reversed(indices_to_remove):
                    del candidates[entity_type]["candidate_spans"][ind]
                    del candidates[entity_type]["candidate_entities"][ind]
                    del candidates[entity_type]["candidate_entity_priors"][ind]

        # get the segment ids for the spans
        for key, cands in candidates.items():
            span_segment_ids = []
            for candidate_span in cands['candidate_spans']:
                span_segment_ids.append(segment_ids[candidate_span[0]])
            candidates[key]['candidate_segment_ids'] = span_segment_ids

        fields['tokens'] = tokens
        fields['segment_ids'] = segment_ids
        fields['candidates'] = candidates
        fields['offsets_a'] = offsets_a
        fields['offsets_b'] = offsets_b
        return fields

    def _tokenize_text(self, text):
        if self.whitespace_tokenize:
            tokens = text.split()
        else:
            tokens = self.bert_word_tokenizer.tokenize(text)

        word_piece_tokens = []
        offsets = [0]
        for token in tokens:
            word_pieces = self._word_to_word_pieces(token)
            offsets.append(offsets[-1] + len(word_pieces))
            word_piece_tokens.append(word_pieces)
        del offsets[0]
        return offsets, word_piece_tokens, tokens

    def _generate_sentence_entity_candidates(self, tokens, offsets):
        """
        Tokenize sentence, trim it to the target length, and generate entity candidates.
        :param sentence
        :param target_length: The length of the output sentence in terms of word pieces.
        :return: Dict[str, Dict[str, Any]],
            {'wordnet': {'candidate_spans': List[List[int]],
                         'candidate_entities': List[List[str]],
                         'candidate_entity_priors': List[List[float]]},
             'wiki': ...}

        """
        assert len(tokens) == len(offsets), f'Length of tokens {len(tokens)} must equal that of offsets {len(offsets)}.'
        entity_instances = {}
        for name, mention_generator in self.candidate_generators.items():
            entity_instances[name] = mention_generator.get_mentions_raw_text(' '.join(tokens), whitespace_tokenize=True)

        for name, entities in entity_instances.items():
            candidate_spans = entities["candidate_spans"]
            adjusted_spans = []
            for start, end in candidate_spans:
                if 0 < start:
                    adjusted_span = [offsets[start - 1], offsets[end] - 1]
                else:
                    adjusted_span = [0, offsets[end] - 1]
                adjusted_spans.append(adjusted_span)
            entities['candidate_spans'] = adjusted_spans
            entity_instances[name] = entities
        return entity_instances

    def convert_tokens_candidates_to_fields(self, tokens_and_candidates):
        """
        tokens_and_candidates is the return from a previous call to
        generate_sentence_entity_candidates.  Converts the dict to
        a dict of fields usable with allennlp.
        """
        fields = {}

        fields['tokens'] = TextField(
                [Token(t, text_id=self.bert_tokenizer.vocab[t])
                    for t in tokens_and_candidates['tokens']],
                token_indexers=self._bert_single_id_indexer
        )

        fields['segment_ids'] = ArrayField(
            np.array(tokens_and_candidates['segment_ids']), dtype=np.int
        )

        all_candidates = {}
        for key, entity_candidates in tokens_and_candidates['candidates'].items():
            # pad the prior to create the array field
            # make a copy to avoid modifying the input
            candidate_entity_prior = copy.deepcopy(
                    entity_candidates['candidate_entity_priors']
            )
            max_cands = max(len(p) for p in candidate_entity_prior)
            for p in candidate_entity_prior:
                if len(p) < max_cands:
                    p.extend([0.0] * (max_cands - len(p)))
            np_prior = np.array(candidate_entity_prior)

            candidate_fields = {
                "candidate_entity_priors": ArrayField(np_prior, dtype=self.dtype),
                "candidate_entities": TextField(
                    [Token(" ".join(candidate_list)) for candidate_list in entity_candidates["candidate_entities"]],
                    token_indexers={'ids': self._entity_indexers[key]}),
                "candidate_spans": ListField(
                    [SpanField(span[0], span[1], fields['tokens']) for span in
                    entity_candidates['candidate_spans']]
                ),
                "candidate_segment_ids": ArrayField(
                    np.array(entity_candidates['candidate_segment_ids']), dtype=np.int
        )
            }
            all_candidates[key] = DictField(candidate_fields)

        fields["candidates"] = DictField(all_candidates)

        return fields





def mask_entities(lm_labels, all_candidate_spans):
    """
    lm_labels = [PAD] where not making a prediction, otherwise the target token
    all_candidate_spans = list of span start/end

    returns spans_to_mask, spans_to_random
        each is a list of span start/end
        spans_to_mask = candidate spans to replace with @@MASK@@
        spans_to_random = candidate_spans to replace with random entities

    For each candidate span that overlaps with a masked location:
        80% we mask out
        10% we keep as is
        10% we replace with random
    """
    masked_indices = [index for index, lm_label in enumerate(lm_labels)
                      if lm_label != '[PAD]']

    spans_to_mask = set()
    spans_to_random = set()
    for index in masked_indices:
        for span in all_candidate_spans:
            if index >= span[0] and index <= span[1]:
                # this candidate has been masked
                if np.random.random() < 0.8:
                    # mask it out
                    spans_to_mask.add(tuple(span))
                else:
                    if np.random.random() < 0.5:
                        # keep as is, do nothing
                        pass
                    else:
                        # random
                        spans_to_random.add(tuple(span))

    return spans_to_mask, spans_to_random



def replace_candidates_with_mask_entity(candidates, spans_to_mask):
    """
    candidates = key -> {'candidate_spans': ...}
    """
    for candidate_key in candidates.keys():
        indices_to_mask = []
        for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
            if tuple(candidate_span) in spans_to_mask:
                indices_to_mask.append(k)
        for ind in indices_to_mask:
            candidates[candidate_key]['candidate_entities'][ind] = ['@@MASK@@']
            candidates[candidate_key]['candidate_entity_priors'][ind] = [1.0]


def replace_candidates_with_random_entity(candidates, spans_to_random):
    for candidate_key in candidates.keys():

        all_entities = list(set(chain.from_iterable(candidates[candidate_key]['candidate_entities'])))

        indices_to_random = []
        for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
            if tuple(candidate_span) in spans_to_random:
                indices_to_random.append(k)

        for ind in indices_to_random:
            n = np.random.randint(5) + 1
            random.shuffle(all_entities)
            rand_entities = all_entities[:n]
            candidates[candidate_key]['candidate_entities'][ind] = list(rand_entities)
            prior = np.random.rand(len(rand_entities))
            prior /= prior.sum()
            candidates[candidate_key]['candidate_entity_priors'][ind] = prior.tolist()



class BertTokenizerCandidateGeneratorMasker:
    """
    Handles:
        * tokenizing sentence_a, sentence_b
        * generating candidates
        * adjust candidate spans for word pieces
        * LM masking
        * interaction of LM masking with candidates
        * converting to fields
    """
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 max_predictions_per_seq: int = 20,
                 masked_lm_prob: float = 0.15,
                 mask_candidate_strategy: str = 'none'):

        self.tokenizer_and_candidate_generator = tokenizer_and_candidate_generator

        self.max_predictions_per_seq = max_predictions_per_seq
        self.masked_lm_prob = masked_lm_prob

        self._label_indexer = {"lm_labels": self.tokenizer_and_candidate_generator._bert_single_id_indexer["tokens"]}
        assert mask_candidate_strategy in ('none', 'full_mask')
        self._mask_candidate_strategy = mask_candidate_strategy

    def tokenize_candidates_mask(self, sentence1: str, sentence2: str):
        """
        # call BertTokenizerAndCandidateGenerator.tokenize_and_generate_candidates
        # call convert_tokens_candidates_to_fields to convert to fields
        # do LM masking, and convert LM masks to fields
        """
        # Generate entity candidates
        token_candidates = self.tokenizer_and_candidate_generator.tokenize_and_generate_candidates(
            sentence1, sentence2
        )

        # LM masking
        masked_tokens, lm_labels = self.create_masked_lm_predictions(
                token_candidates['tokens']
        )

        # masking interaction with spans
        if self._mask_candidate_strategy == 'full_mask':
            all_candidate_spans = []
            for key in token_candidates['candidates'].keys():
                all_candidate_spans.extend(
                        token_candidates['candidates'][key]['candidate_spans']
                )

            spans_to_mask, spans_to_random = mask_entities(lm_labels, all_candidate_spans)
            replace_candidates_with_mask_entity(
                    token_candidates['candidates'], spans_to_mask
            )
            replace_candidates_with_random_entity(
                    token_candidates['candidates'], spans_to_random
            )

        token_candidates['tokens'] = masked_tokens

        # Converting to fields
        fields = self.tokenizer_and_candidate_generator. \
            convert_tokens_candidates_to_fields(token_candidates)

        # Adding LM labels field
        fields['lm_label_ids'] = TextField(
            [Token(t, text_id=self.tokenizer_and_candidate_generator.bert_tokenizer.vocab[t]) for t in lm_labels],
            token_indexers=self._label_indexer
        )

        return fields


    def create_masked_lm_predictions(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """Creates the predictions for the masked LM objective.
           Assumes tokens is already word piece tokenized and truncated"""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == start_token or token == sep_token:
                continue
            cand_indexes.append(i)

        np.random.shuffle(cand_indexes)

        # the return list of tokens, with [MASK]
        output_tokens = list(tokens)

        num_to_predict = min(self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob)))
        )

        lm_labels = ["[PAD]"] * len(tokens)
        vocab = self.tokenizer_and_candidate_generator.bert_tokenizer.ids_to_tokens

        for index in cand_indexes[:num_to_predict]:
            # 80% of the time, replace with [MASK]
            if np.random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if np.random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab[np.random.randint(0, len(vocab))]

            output_tokens[index] = masked_token

            lm_labels[index] = tokens[index]

        return output_tokens, lm_labels






@DataIterator.register("cross_sentence_linking")
class CrossSentenceLinking(DataIterator):
    """
    Assumes the dataset reader is not lazy
    """
    def __init__(self,
                 batch_size: int,
                 entity_indexer: TokenIndexer,
                 bert_model_type: str,
                 do_lower_case: bool,
                 mask_candidate_strategy: str = 'none',
                 dataset_index: int = None,
                 iterate_forever: bool = False,
                 id_type: str = 'wordnet',
                 max_predictions_per_seq: int = 20,
                 use_nsp_label: bool = True,
                 max_word_piece_sequence_length: int = 512,
                 extra_id_type: str = None,
                 extra_entity_indexer: TokenIndexer = None):

        super().__init__(batch_size)

        self.batch_size = batch_size
        self.iterate_forever = iterate_forever
        self.id_type = id_type
        self.entity_indexer = {"ids": entity_indexer}
        self.dataset_index = dataset_index
        self.use_nsp_label = use_nsp_label

        assert id_type in ('wordnet', 'wiki')
        entity_candidate_generators = {self.id_type: StubbedCandidateGenerator()}
        entity_indexers = {self.id_type: entity_indexer}

        self.extra_id_type = extra_id_type
        self.extra_entity_indexer = {"ids": extra_entity_indexer}
        if self.extra_id_type is not None:
            assert self.extra_id_type in ('wordnet', 'wiki')
            entity_candidate_generators[self.extra_id_type] = StubbedCandidateGenerator()
            entity_indexers[self.extra_id_type] = extra_entity_indexer

        # we will use an instance of BertTokenizerCandidateGeneratorMasker
        # to combine the instances and tokenize and mask
        tokenizer_and_candidate_generator = BertTokenizerAndCandidateGenerator(
            entity_candidate_generators=entity_candidate_generators,
            entity_indexers=entity_indexers,
            bert_model_type=bert_model_type,
            do_lower_case=do_lower_case,
            whitespace_tokenize=True,
            max_word_piece_sequence_length=max_word_piece_sequence_length
        )

        self.tokenizer_and_masker = BertTokenizerCandidateGeneratorMasker(
            tokenizer_and_candidate_generator,
            mask_candidate_strategy=mask_candidate_strategy,
            max_predictions_per_seq=max_predictions_per_seq
        )


    def _get_document_id_wordnet(self, instance):
        gold_key_ids = instance['gold_data_ids']
        if gold_key_ids[0][:18] == 'example_definition':
            # key is the full thing like 'example_definition.55'
            document_key = gold_key_ids[0]
        else:
            # key is d000.s000.t000
            document_key = gold_key_ids[0].partition('.')[0]
        return document_key

    def _get_document_id_aida(self, instance):
        return instance['doc_id'].metadata

    def _group_instances_into_documents(self, instances):
        # we need negative samples for each document, so make dict
        # documents, where each document is a list of sentences
        documents = {}
        for instance in instances:
            if self.id_type == 'wordnet':
                document_key = self._get_document_id_wordnet(instance)
            elif self.id_type == 'wiki':
                document_key = self._get_document_id_aida(instance)
            if document_key not in documents:
                documents[document_key] = []
            documents[document_key].append(instance)
        return documents

    def _set_entity_candidate_generator_cache(self, instances):
        # set the cache on the entity candidate generator
        # also keep track of the gold entities.  some text is duplicated
        # with inconsistent annotations, so we'll make the assumption
        # to just keep the first annotations for each unique text string.
        cache = {}
        extra_cache = {}
        gold_cache = {}
        for instance in instances:
            text = ' '.join([t.text for t in instance['tokens'].tokens])
            if text not in cache:
                candidate_spans = [
                    [span.span_start, span.span_end] for span in instance['candidate_spans']
                ]
                candidate_entities = [
                    t.text.split() for t in instance.fields['candidate_entities'].tokens
                ]
                candidate_entity_prior = [
                    p[p > 0].tolist()
                    for p in instance['candidate_entity_prior'].array
                ]
                candidate_segment_ids = instance.fields['candidate_segment_ids'].array.tolist()

                is_sorted = sorted(candidate_spans) == candidate_spans
                if not is_sorted:
                    sort_indices = [ele[0] for ele in sorted(enumerate(candidate_spans), key=lambda x: x[1])]
                else:
                    sort_indices = list(range(len(candidate_spans)))

                sorted_candidate_spans = [candidate_spans[i] for i in sort_indices]
                sorted_candidate_entities = [candidate_entities[i] for i in sort_indices]
                sorted_candidate_entity_prior = [candidate_entity_prior[i] for i in sort_indices]
                sorted_candidate_segment_ids = [candidate_segment_ids[i] for i in sort_indices]
                candidate = {'candidate_spans': sorted_candidate_spans,
                             'candidate_entities': sorted_candidate_entities,
                             'candidate_entity_priors': sorted_candidate_entity_prior,
                             'candidate_segment_ids': sorted_candidate_segment_ids}

                cache[text] = candidate

                gold_cache[text] = copy.deepcopy(
                        [instance['gold_entities'].tokens[i].text for i in sort_indices]
                )

                # extra candidate generator
                if self.extra_id_type is not None:
                    extra_candidates = instance['extra_candidates'].metadata[self.extra_id_type]
                    e_candidate = {
                            'candidate_spans':  extra_candidates['candidate_spans'],
                            'candidate_entities': extra_candidates['candidate_entities'],
                            'candidate_entity_priors': extra_candidates['candidate_entity_priors'],
                            'candidate_segment_ids': [0] * len(extra_candidates['candidate_spans']),
                    }
                    extra_cache[text] = e_candidate

        self.tokenizer_and_masker.tokenizer_and_candidate_generator.candidate_generators[self.id_type].set_cache(cache)
        if self.extra_id_type is not None:
            self.tokenizer_and_masker.tokenizer_and_candidate_generator.candidate_generators[self.extra_id_type].set_cache(extra_cache)

        return gold_cache

    def _combine_instances(self, instance_a, instance_b, nsp_label, gold_cache):
        text_a = ' '.join([t.text for t in instance_a['tokens'].tokens])
        text_b = ' '.join([t.text for t in instance_b['tokens'].tokens])

        fields = self.tokenizer_and_masker.tokenize_candidates_mask(text_a, text_b)
        candidate_spans = [
            [s.span_start, s.span_end]
            for s in fields['candidates'].field_dict[self.id_type].field_dict['candidate_spans'].field_list
        ]
        assert sorted(candidate_spans) == candidate_spans

        # combine the gold entities
        golds = []
        for text in [text_a, text_b]:
            golds.append(gold_cache[text])

        combined_golds = []
        j = [-1, -1]
        for span in candidate_spans:
            i = fields['segment_ids'].array[span[0]]
            j[i] += 1
            combined_golds.append(golds[i][j[i]])

        gold_text_field = TextField(
            [Token(g) for g in combined_golds],
            token_indexers=self.entity_indexer
        )
        fields['gold_entities'] = DictField({self.id_type: gold_text_field})

        if self.use_nsp_label:
            fields['next_sentence_label'] = LabelField(nsp_label, skip_indexing=True)

        del fields['lm_label_ids']

        return Instance(fields)


    def _create_batches(self, instances, shuffle: bool = True):
        if self.iterate_forever:
            num_epochs = 100000000
        else:
            num_epochs = 1

        documents = self._group_instances_into_documents(instances)
        document_keys = list(documents.keys())

        # set the cache on the entity candidate generator
        gold_cache = self._set_entity_candidate_generator_cache(instances)

        for epoch_num in range(num_epochs):
            new_instances = []
            for document_key, document_instances in documents.items():
                for k in range(len(document_instances)):
                    if k == len(document_instances) - 1 or random.random() < 0.5:
                        for _ in range(10):
                            random_key = random.choice(document_keys)
                            if random_key != document_key:
                                break

                        random_doc = documents[random_key]
                        instance_b = random.choice(random_doc)
                        nsp_label = 1

                    else:
                        # actual next
                        instance_b = document_instances[k + 1]
                        nsp_label = 0

                    instance_a = document_instances[k]

                    new_instances.append(
                        self._combine_instances(instance_a, instance_b, nsp_label, gold_cache)
                    )

            random.shuffle(new_instances)

            start = 0
            while start < len(new_instances):
                end = start + self.batch_size
                batch = Batch(new_instances[start:end])
                yield Batch(new_instances[start:end])
                start = end

    def __call__(self, *args, **kwargs):
        for batch in super().__call__(*args, **kwargs):
            if self.dataset_index is not None:
                batch['dataset_index'] = torch.tensor(self.dataset_index)
            yield batch






if __name__ == "__main__":
    reader = LinkingReader()





























