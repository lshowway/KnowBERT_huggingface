from typing import List, Tuple, Union
import time
import json
import random

import spacy
from spacy.lang.en import STOP_WORDS
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY
from spacy.tokens import Doc


from allennlp.common.registrable import Registrable


def prior_entity_candidates(candidates_file: str,
                            max_candidates:int = 30,
                            allowed_entities_set=None,
                            max_mentions = None):
    """
    Args:
    cand_ent_num: how many candidate entities to keep for each mention
    allowed_entities_set: restrict the candidate entities to only this set. for example
    the most frequent 1M entities. First this restiction applies and then the cand_ent_num.
    """
    wall_start = time.time()
    p_e_m = {}  # for each mention we have a list of tuples (ent_id, score)
    mention_total_freq = {}  # for each mention of the p_e_m we store the total freq
                                # this will help us decide which cand entities to take
    p_e_m_errors = 0
    incompatible_ent_ids = 0
    duplicate_mentions_cnt = 0
    clear_conflict_winner = 0  # both higher absolute frequency and longer cand list
    not_clear_conflict_winner = 0  # higher absolute freq but shorter cand list
    with open(candidates_file, encoding='utf-8') as fin:

        for i, line in enumerate(fin):

            if max_mentions is not None and i > max_mentions:
                break
            line = line.rstrip()
            line_parts = line.split("\t")
            mention = line_parts[0]
            absolute_freq = int(line_parts[1])
            entities = line_parts[2:]
            entity_candidates = []
            for e in entities:
                if len(entity_candidates) >= max_candidates:
                    break
                ent_id, score, name = [x.strip() for x in e.split(',', 2)]
                if allowed_entities_set is not None and name not in allowed_entities_set:
                    pass
                else:
                    entity_candidates.append((ent_id, name, float(score)))
            if entity_candidates:
                if mention in p_e_m:
                    duplicate_mentions_cnt += 1
                    #print("duplicate mention: ", mention)
                    if absolute_freq > mention_total_freq[mention]:
                        if len(entity_candidates) > len(p_e_m[mention]):
                            clear_conflict_winner += 1
                        else:
                            not_clear_conflict_winner += 1
                        p_e_m[mention] = entity_candidates
                        mention_total_freq[mention] = absolute_freq
                else:
                    # for each mention we have a list of tuples (ent_id, name, score)
                    p_e_m[mention] = entity_candidates
                    mention_total_freq[mention] = absolute_freq

    print("duplicate_mentions_cnt: ", duplicate_mentions_cnt)
    print("end of p_e_m reading. wall time:", (time.time() - wall_start)/60, " minutes")
    print("p_e_m_errors: ", p_e_m_errors)
    print("incompatible_ent_ids: ", incompatible_ent_ids)

    wall_start = time.time()
    # two different p(e|m) mentions can be the same after lower() so we merge the two candidate
    # entities lists. But the two lists can have the same candidate entity with different score
    # we keep the highest score. For example if "Obama" mention gives 0.9 to entity Obama and
    # OBAMA gives 0.7 then we keep the 0.9 . Also we keep as before only the cand_ent_num entities
    # with the highest score

    p_e_m_lowercased = {}
    for mention, candidates in p_e_m.items(): # 767w
        l_mention = mention.lower()
        lower_candidates = p_e_m.get(l_mention, [])
        combined_candidates = {}

        for cand in candidates + lower_candidates:
            if cand[0] in combined_candidates:
                if cand[2] > combined_candidates[cand[0]][2]:
                    combined_candidates[0] = cand

            else:
                combined_candidates[cand[0]] = cand

        combined_candidates = list(combined_candidates.values())
        sorted_candidates = sorted(combined_candidates, key=lambda x: x[2], reverse=True)

        p_e_m_lowercased[l_mention] = sorted_candidates[:max_candidates]

    return p_e_m, p_e_m_lowercased, mention_total_freq



STOP_SYMBOLS = set().union(LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY)
def span_filter_func(span: List[str]):
    """
    This function halves the number of suggested mention spans whilst not affecting
    gold span recall at all. It can probably be improved further.
    """
    if span[0] in STOP_WORDS or span[-1] in STOP_WORDS:
        return False

    if any([c in STOP_SYMBOLS for c in span]):
        return False
    return True


# from https://spacy.io/usage/linguistic-features#custom-tokenizer-example
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)



def get_empty_candidates():
    """
    The mention generators always return at least one candidate, but signal
    it with this special candidate
    """
    return {
        "candidate_spans": [[-1, -1]],
        "candidate_entities": [["@@PADDING@@"]],
        "candidate_entity_priors": [[1.0]]
    }


class MentionGenerator(Registrable):
    pass