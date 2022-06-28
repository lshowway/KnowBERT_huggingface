from typing import Dict

import math
import tarfile
from overrides import overrides
import logging


from pytorch_pretrained_bert.modeling import BertForPreTraining, BertLayerNorm, BertConfig, BertEncoder
from pytorch_pretrained_bert.modeling import BertIntermediate, BertOutput, BertSelfOutput

import torch
from torch import nn
import numpy as np
from torch.nn.functional import embedding
from allennlp.common.file_utils import cached_path
from allennlp.modules.span_extractors.span_extractor import SpanExtractor

from utils import F1Metric, combine_initial_dims, uncombine_initial_dims, TimeDistributed

from utils import init_bert_weights, extend_attention_mask_for_bert, get_dtype_for_module

import utils

from metrics import ExponentialMovingAverage, Average, WeightedAverage, CategoricalAccuracy, MeanReciprocalRank

import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)



def print_shapes(x, prefix='', raise_on_nan=False):
    if isinstance(x, torch.Tensor):
        print(prefix, x.shape)
        if x.dtype == torch.float32 or x.dtype == torch.float16:
            print(x.min(), x.max(), x.mean(), x.std())
        if raise_on_nan and torch.isnan(x).long().sum().item() > 0:
            print("GOT NAN!!")
            raise ValueError
    elif isinstance(x, (list, tuple)):
        for ele in x:
            print_shapes(ele, prefix + '-->')
    elif isinstance(x, dict):
        for k, v in x.items():
            print_shapes(v, prefix + ' ' + k + ':')
    else:
        print("COULDN'T get shape ", type(x))


def diagnose_backward_hook(module, m_input, m_output):
    print("------")
    print('Inside ' + module.__class__.__name__ + ' backward')
    print('Inside class:' + module.__class__.__name__)
    print("INPUT:")
    print_shapes(m_input)
    print("OUTPUT:")
    print_shapes(m_output)
    print("=======")


def diagnose_forward_hook(module, m_input, m_output):
    print("------")
    print('Inside ' + module.__class__.__name__ + ' forward')
    print('Inside class:' + module.__class__.__name__)
    print("INPUT:")
    print_shapes(m_input)
    print("OUTPUT:")
    print_shapes(m_output, raise_on_nan=True)
    print("=======")


# ============== 1. from EntityLinkingWithCandidateMentions



class DotAttentionWithPrior(torch.nn.Module):
    def __init__(self,
                 output_feed_forward_hidden_dim: int = 100,
                 weighted_entity_threshold: float = None,
                 null_embedding: torch.Tensor = None,
                 initializer_range: float = 0.02):

        super().__init__()

        # layers for the dot product attention
        self.out_layer_1 = torch.nn.Linear(2, output_feed_forward_hidden_dim)
        self.out_layer_2 = torch.nn.Linear(output_feed_forward_hidden_dim, 1)
        init_bert_weights(self.out_layer_1, initializer_range)
        init_bert_weights(self.out_layer_2, initializer_range)

        self.weighted_entity_threshold = weighted_entity_threshold
        if null_embedding is not None:
            self.register_buffer('null_embedding', null_embedding)

    def forward(self,
            projected_span_representations,
            candidate_entity_embeddings,
            candidate_entity_prior,
            entity_mask):
        """
        projected_span_representations = (batch_size, num_spans, entity_dim)
        candidate_entity_embeddings = (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_prior = (batch_size, num_spans, num_candidates)
            with prior probability of each candidate entity.
            0 <= candidate_entity_prior <= 1 and candidate_entity_prior.sum(dim=-1) == 1
        entity_mask = (batch_size, num_spans, num_candidates)
            with 0/1 bool of whether it is a valid candidate

        returns dict with:
            linking_scores: linking sccore to each entity in each span
                (batch_size, num_spans, num_candidates)
                masked with -10000 for invalid links
            weighted_entity_embeddings: weighted entity embedding
                (batch_size, num_spans, entity_dim)
        """
        # dot product between span embedding and entity embeddings, scaled
        # by sqrt(dimension) as in Transformer
        # (batch_size, num_spans, num_candidates)
        scores = torch.sum(
            projected_span_representations.unsqueeze(-2) * candidate_entity_embeddings,
            dim=-1
        ) / math.sqrt(candidate_entity_embeddings.shape[-1])

        # compute the final score
        # the prior needs to be input as float32 due to half not supported on
        # cpu.  so need to cast it here.
        dtype = list(self.parameters())[0].dtype
        scores_with_prior = torch.cat(
            [scores.unsqueeze(-1), candidate_entity_prior.unsqueeze(-1).to(dtype)],
            dim=-1
        )

        # (batch_size, num_spans, num_candidates)
        linking_score = self.out_layer_2(
            torch.nn.functional.relu(self.out_layer_1(scores_with_prior))
        ).squeeze(-1)

        # mask out the invalid candidates
        invalid_candidate_mask = ~entity_mask

        linking_scores = linking_score.masked_fill(invalid_candidate_mask.bool(), -10000.0)
        return_dict = {'linking_scores': linking_scores}

        weighted_entity_embeddings = self._get_weighted_entity_embeddings(
                linking_scores, candidate_entity_embeddings
        )
        return_dict['weighted_entity_embeddings'] = weighted_entity_embeddings

        return return_dict


    def _get_weighted_entity_embeddings(self, linking_scores, candidate_entity_embeddings):
        """
        Get the entity linking weighted entity embedding

        linking_scores = (batch_size, num_spans, num_candidates)
             with unnormalized scores and masked with very small value
            (-10000) for invalid candidates.
        candidate_entity_embeddings = (batch_size, num_spans, num_candidates, entity_embedding_dim)

        returns weighted_entity_embeddings = (batch_size, num_spans, entity_dim)
        """
        # compute softmax of linking scores
        # if we are using the decode threshold, set all scores less then
        # the threshold to small values so they aren't given any weight
        if self.weighted_entity_threshold is not None:
            below_threshold = linking_scores < self.weighted_entity_threshold
            linking_scores = linking_scores.masked_fill(below_threshold.bool(), -10000.0)

        # (batch_size, num_spans, num_candidates)
        normed_linking_scores = torch.nn.functional.softmax(linking_scores, dim=-1)

        # use softmax to get weighted entity embedding from candidates
        # (batch_size, num_spans, entity_dim)
        weighted_entity_embeddings = (normed_linking_scores.unsqueeze(-1) * candidate_entity_embeddings).sum(dim=2)

        # if we have a decode threshold, some spans won't have a single
        # predicted entity above the threshold, need to replace them with
        # NULL
        if self.weighted_entity_threshold is not None:
            num_candidates = linking_scores.shape[-1]
            # (batch_size, num_spans)
            all_below_threshold = (below_threshold == 1).long().sum(dim=-1) == num_candidates
            weighted_entity_embeddings[all_below_threshold] = self.null_embedding

        return weighted_entity_embeddings



class SelfAttentiveSpanExtractor(SpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.

    Returns
    -------
    attended_text_embeddings : ``torch.FloatTensor``.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """
    def __init__(self,
                 input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:

        dtype = sequence_tensor.dtype

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = utils.get_range_vector(max_batch_span_width,
                                                       utils.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).to(dtype)
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).to(dtype)
        span_indices = torch.nn.functional.relu(raw_span_indices.to(dtype)).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = utils.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = utils.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = utils.batched_index_select(global_attention_logits,
                                                          span_indices,
                                                          flat_span_indices).squeeze(-1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = utils.masked_softmax(span_attention_logits, span_mask,
                                                     memory_efficient=True,
                                                     mask_fill_value=-1000)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = utils.weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).to(dtype)

        return attended_text_embeddings



class EntityDisambiguator(nn.Module):
    def __init__(self,
                 contextual_embedding_dim,
                 entity_embedding_dim: int,
                 entity_embeddings: torch.nn.Embedding,
                 max_sequence_length: int = 512,
                 span_encoder_config: Dict[str, int] = None,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 weighted_entity_threshold: float = None,
                 null_entity_id: int = None,
                 include_null_embedding_in_dot_attention: bool = False):
        """
        Idea: Align the bert and KG vector space by learning a mapping between
            them.
        """
        super().__init__()

        self.span_extractor = SelfAttentiveSpanExtractor(entity_embedding_dim)
        init_bert_weights(self.span_extractor._global_attention._module,
                          initializer_range)

        self.dropout = torch.nn.Dropout(dropout)

        self.bert_to_kg_projector = torch.nn.Linear(
                contextual_embedding_dim, entity_embedding_dim)
        init_bert_weights(self.bert_to_kg_projector, initializer_range)
        self.projected_span_layer_norm = BertLayerNorm(entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.projected_span_layer_norm, initializer_range)

        self.kg_layer_norm = BertLayerNorm(entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.kg_layer_norm, initializer_range)

        # already pretrained, don't init
        self.entity_embeddings = entity_embeddings
        self.entity_embedding_dim = entity_embedding_dim

        # layers for the dot product attention
        if weighted_entity_threshold is not None or include_null_embedding_in_dot_attention:
            if hasattr(self.entity_embeddings, 'get_null_embedding'):
                null_embedding = self.entity_embeddings.get_null_embedding()
            else:
                null_embedding = self.entity_embeddings.weight[null_entity_id, :]
        else:
            null_embedding = None
        self.dot_attention_with_prior = DotAttentionWithPrior(
                 output_feed_forward_hidden_dim,
                 weighted_entity_threshold,
                 null_embedding,
                 initializer_range
        )
        self.null_entity_id = null_entity_id
        self.contextual_embedding_dim = contextual_embedding_dim

        if span_encoder_config is None:
            self.span_encoder = None
        else:
            # create BertConfig
            assert len(span_encoder_config) == 4
            config = BertConfig(
                0, # vocab size, not used
                hidden_size=span_encoder_config['hidden_size'],
                num_hidden_layers=span_encoder_config['num_hidden_layers'],
                num_attention_heads=span_encoder_config['num_attention_heads'],
                intermediate_size=span_encoder_config['intermediate_size']
            )
            self.span_encoder = BertEncoder(config)
            init_bert_weights(self.span_encoder, initializer_range)


    def unfreeze(self, mode):
        def _is_in_alignment(n):
            if 'bert_to_kg_projector' in n:
                return True
            elif 'projected_span_layer_norm' in n:
                return True
            elif 'kg_position_embeddings.embedding_projection' in n:
                return True
            elif 'kg_position_embeddings.position_layer_norm' in n:
                return True
            elif 'kg_layer_norm' in n:
                return True
            elif 'span_extractor' in n:
                return True
            else:
                return False

        if mode == 'entity_linking':
            # learning the entity linker
            for n, p in self.named_parameters():
                if _is_in_alignment(n):
                    p.requires_grad_(True)
                elif 'entity_embeddings.weight' in n:
                    p.requires_grad_(False)
                elif 'kg_position_embeddings' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
        elif mode == 'freeze':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for n, p in self.named_parameters():
                if 'entity_embeddings.weight' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)

    def _run_span_encoders(self, x, span_mask):
        # run the transformer
        attention_mask = extend_attention_mask_for_bert(span_mask, get_dtype_for_module(self))
        return self.span_encoder(
            x, attention_mask,
            output_all_encoded_layers=False
        )

    def forward(self,
                contextual_embeddings: torch.Tensor,
                mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                gold_entities
        ):
        # get the candidate entity embeddings
        # (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_embeddings = self.entity_embeddings(candidate_entities)
        candidate_entity_embeddings = self.kg_layer_norm(candidate_entity_embeddings.contiguous())

        # project to entity embedding dim
        # (batch_size, timesteps, entity_dim)
        projected_bert_representations = self.bert_to_kg_projector(contextual_embeddings)

        # compute span representations
        span_mask = (candidate_spans[:, :, 0] > -1).long()
        # (batch_size, num_spans, embedding_dim)
        projected_span_representations = self.span_extractor(
            projected_bert_representations,
            candidate_spans,
            mask,
            span_mask
        )
        projected_span_representations = \
            self.projected_span_layer_norm(projected_span_representations.contiguous())

        # run the span transformer encoders
        if self.span_encoder is not None:
            projected_span_representations = \
                self._run_span_encoders(
                projected_span_representations, span_mask
            )[-1] # 16*47*300

        entity_mask = candidate_entities > 0
        return_dict = self.dot_attention_with_prior(
                    projected_span_representations,
                    candidate_entity_embeddings,
                    candidate_entity_priors,
                    entity_mask)

        return_dict['projected_span_representations'] = projected_span_representations
        return_dict['projected_bert_representations'] = projected_bert_representations

        return return_dict



class EntityLinkingBase(nn.Module):
    def __init__(self,
                 margin=0.2,
                 decode_threshold=0.0,
                 loss_type='margin',
                 namespace='entity',
                 ):

        super().__init__()

        if loss_type == 'margin':
            self.loss = nn.MarginRankingLoss(margin=margin)
            self.decode_threshold = decode_threshold
        elif loss_type == 'softmax':
            self.loss = nn.NLLLoss(ignore_index=-100)
            # set threshold to small value so we just take argmax
            self._log_softmax = nn.LogSoftmax(dim=-1)
            self.decode_threshold = -990
        else:
            raise ValueError("invalid loss type, got {}".format(loss_type))
        self.loss_type = loss_type

        self.null_entity_id = 470115
        self.unk_entity_id = 1

        self._f1_metric = F1Metric()
        self._f1_metric_untyped = F1Metric()


    def _compute_f1(self, linking_scores, candidate_spans,
                    candidate_entities, gold_entities):
        # will call F1Metric with predicted and gold entities encoded as
        # [(start, end), entity_id]

        predicted_entities = self._decode(
                    linking_scores, candidate_spans, candidate_entities
        )

        # make a mask of valid predictions and non-null entities to select
        # ids and spans
        # (batch_size, num_spans, 1)
        gold_mask = (gold_entities > 0) & (gold_entities != self.null_entity_id)

        valid_gold_entity_spans = candidate_spans[
                torch.cat([gold_mask, gold_mask], dim=-1)
        ].view(-1, 2).tolist()
        valid_gold_entity_id = gold_entities[gold_mask].tolist()

        batch_size, num_spans, _ = linking_scores.shape
        batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat([1, num_spans])[gold_mask.squeeze(-1).cpu()]

        gold_entities_for_f1 = []
        predicted_entities_for_f1 = []
        gold_spans_for_f1 = []
        predicted_spans_for_f1 = []
        for k in range(batch_size):
            gold_entities_for_f1.append([])
            predicted_entities_for_f1.append([])
            gold_spans_for_f1.append([])
            predicted_spans_for_f1.append([])

        for gi, gs, g_batch_index in zip(valid_gold_entity_id,
                              valid_gold_entity_spans,
                              batch_indices.tolist()):
            gold_entities_for_f1[g_batch_index].append((tuple(gs), gi))
            gold_spans_for_f1[g_batch_index].append((tuple(gs), "ENT"))

        for p_batch_index, ps, pi in predicted_entities:
            span = tuple(ps)
            predicted_entities_for_f1[p_batch_index].append((span, pi))
            predicted_spans_for_f1[p_batch_index].append((span, "ENT"))

        self._f1_metric_untyped(predicted_spans_for_f1, gold_spans_for_f1)
        self._f1_metric(predicted_entities_for_f1, gold_entities_for_f1)


    def _decode(self, linking_scores, candidate_spans, candidate_entities):
        # returns [[batch_index1, (start1, end1), eid1],
        #          [batch_index2, (start2, end2), eid2], ...]

        # Note: We assume that linking_scores has already had the mask
        # applied such that invalid candidates have very low score. As a result,
        # we don't need to worry about masking the valid candidate spans
        # here, since their score will be very low, and won't exceed
        # the threshold.

        # find maximum candidate entity score in each valid span
        # (batch_size, num_spans), (batch_size, num_spans)
        max_candidate_score, max_candidate_indices = linking_scores.max(dim=-1)

        # get those above the threshold
        above_threshold_mask = max_candidate_score > self.decode_threshold

        # for entities with score > threshold:
        #       get original candidate span
        #       get original entity id
        # (num_extracted_spans, 2)
        extracted_candidates = candidate_spans[above_threshold_mask]
        # (num_extracted_spans, num_candidates)
        candidate_entities_for_extracted_spans = candidate_entities[above_threshold_mask]
        extracted_indices = max_candidate_indices[above_threshold_mask]
        # the batch number (num_extracted_spans, )
        batch_size, num_spans, _ = linking_scores.shape
        batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat([1, num_spans])[above_threshold_mask.cpu()]

        extracted_entity_ids = []
        for k, ind in enumerate(extracted_indices):
            extracted_entity_ids.append(candidate_entities_for_extracted_spans[k, ind])

        # make tuples [(span start, span end), id], ignoring the null entity
        ret = []
        for start_end, eid, batch_index in zip(
                    extracted_candidates.tolist(),
                    extracted_entity_ids,
                    batch_indices.tolist()
        ):
            entity_id = eid.item()
            if entity_id != self.null_entity_id:
                ret.append((batch_index, tuple(start_end), entity_id))

        return ret


    def get_metrics(self, reset: bool = False):
        precision, recall, f1_measure = self._f1_metric.get_metric(reset)
        precision_span, recall_span, f1_measure_span = self._f1_metric_untyped.get_metric(reset)
        metrics = {
            'el_precision': precision,
            'el_recall': recall,
            'el_f1': f1_measure,
            'span_precision': precision_span,
            'span_recall': recall_span,
            'span_f1': f1_measure_span
        }

        return metrics


    def _compute_loss(self, candidate_entities, candidate_spans,
                      linking_scores, gold_entities):

        if self.loss_type == 'margin':
            return self._compute_margin_loss(
                    candidate_entities, candidate_spans, linking_scores, gold_entities
            )
        elif self.loss_type == 'softmax':
            return self._compute_softmax_loss(
                    candidate_entities, candidate_spans, linking_scores, gold_entities
            )


    def _compute_margin_loss(self, candidate_entities, candidate_spans,
                             linking_scores, gold_entities):

        # compute loss
        # in End-to-End Neural Entity Linking
        # loss = max(0, gamma - score) if gold mention
        # loss = max(0, score) if not gold mention
        #
        # torch.nn.MaxMarginLoss(x1, x2, y) = max(0, -y * (x1 - x2) + gamma)
        #   = max(0, -x1 + x2 + gamma)  y = +1
        #   = max(0, gamma - x1) if x2 == 0, y=+1
        #
        #   = max(0, x1 - gamma) if y==-1, x2=0

        candidate_mask = candidate_entities > 0
        # (num_entities, )
        non_masked_scores = linking_scores[candidate_mask]

        # broadcast gold ids to all candidates
        num_candidates = candidate_mask.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )
        # compute +1 / -1 labels for whether each candidate is gold
        positive_labels = (broadcast_gold_entities == candidate_entities).long()
        negative_labels = (broadcast_gold_entities != candidate_entities).long()
        labels = (positive_labels - negative_labels).to(dtype=get_dtype_for_module(self))
        # finally select the non-masked candidates
        # (num_entities, ) with +1 / -1
        non_masked_labels = labels[candidate_mask]

        loss = self.loss(
                non_masked_scores, torch.zeros_like(non_masked_labels),
                non_masked_labels
        )

        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)

        return {'loss': loss}

    def _compute_softmax_loss(self,
                             candidate_entities,
                             candidate_spans,
                             linking_scores,
                             gold_entities):

        # compute log softmax
        # linking scores is already masked with -1000 in invalid locations
        # (batch_size, num_spans, max_num_candidates)
        log_prob = self._log_softmax(linking_scores)

        # get the valid scores.
        # needs to be index into the last time of log_prob, with -100
        # for missing values
        num_candidates = log_prob.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )

        # location of the positive label
        positive_labels = (broadcast_gold_entities == candidate_entities).long()
        # index of the positive class
        targets = positive_labels.argmax(dim=-1)

        # fill in the ignore class
        # DANGER: we assume that each instance has exactly one gold
        # label, and that padded instances are ones for which all
        # candidates are invalid
        # (batch_size, num_spans)
        invalid_prediction_mask = (
            candidate_entities != 0
        ).long().sum(dim=-1) == 0
        targets[invalid_prediction_mask] = -100

        loss = self.loss(log_prob.view(-1, num_candidates), targets.view(-1, ))

        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)

        return {'loss': loss}



class Embedding(nn.Module):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
        5. build all of this easily ``from_params``

    Note that if you are using our data API and are trying to embed a
    :class:`~allennlp.data.fields.TextField`, you should use a
    :class:`~allennlp.modules.TextFieldEmbedder` instead of using this directly.

    Parameters
    ----------
    num_embeddings : int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    projection_dim : int, (optional, default=None)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if ``trainable`` is ``False``.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2):
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False):
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False):
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : str, (optional, default=None):
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.

    Returns
    -------
    An Embedding module.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 vocab_namespace: str = None,
                 pretrained_file: str = None) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise Exception("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None


    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = inputs.size() # 16*47*30
        inputs = combine_initial_dims(inputs) # 752*30

        embedded = embedding(inputs, self.weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)

        # Now (if necessary) add back in the extra dimensions.
        embedded = uncombine_initial_dims(embedded, original_size) # 16*47*30*300

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    def extend_vocab(self,  # pylint: disable=arguments-differ
                     extended_vocab,
                     vocab_namespace: str = None,
                     extension_pretrained_file: str = None,
                     model_path: str = None):
        """
        Extends the embedding matrix according to the extended vocabulary.
        If extension_pretrained_file is available, it will be used for initializing the new words
        embeddings in the extended vocabulary; otherwise we will check if _pretrained_file attribute
        is already available. If none is available, they will be initialized with xavier uniform.

        Parameters
        ----------
        extended_vocab : Vocabulary:
            Vocabulary extended from original vocabulary used to construct
            this ``Embedding``.
        vocab_namespace : str, (optional, default=None)
            In case you know what vocab_namespace should be used for extension, you
            can pass it. If not passed, it will check if vocab_namespace used at the
            time of ``Embedding`` construction is available. If so, this namespace
            will be used or else extend_vocab will be a no-op.
        extension_pretrained_file : str, (optional, default=None)
            A file containing pretrained embeddings can be specified here. It can be
            the path to a local file or an URL of a (cached) remote file. Check format
            details in ``from_params`` of ``Embedding`` class.
        model_path : str, (optional, default=None)
            Path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens". This is only useful
            to give helpful error message when extend_vocab is implicitly called
            by fine-tune or any other command.
        """
        # Caveat: For allennlp v0.8.1 and below, we weren't storing vocab_namespace as an attribute,
        # knowing which is necessary at time of embedding vocab extension. So old archive models are
        # currently unextendable.

        vocab_namespace = vocab_namespace or self._vocab_namespace
        if not vocab_namespace:
            # It's not safe to default to "tokens" or any other namespace.
            logging.info("Loading a model trained before embedding extension was implemented; "
                         "pass an explicit vocab namespace if you want to extend the vocabulary.")
            return

        extended_num_embeddings = extended_vocab.get_vocab_size(vocab_namespace)
        if extended_num_embeddings == self.num_embeddings:
            # It's already been extended. No need to initialize / read pretrained file in first place (no-op)
            return

        if extended_num_embeddings < self.num_embeddings:
            raise Exception(f"Size of namespace, {vocab_namespace} for extended_vocab is smaller than "
                                     f"embedding. You likely passed incorrect vocab or namespace for extension.")

        # Case 1: user passed extension_pretrained_file and it's available.
        if extension_pretrained_file and utils.is_url_or_existing_file(extension_pretrained_file):
            # Don't have to do anything here, this is the happy case.
            pass
        # Case 2: user passed extension_pretrained_file and it's not available
        elif extension_pretrained_file:
            raise Exception(f"You passed pretrained embedding file {extension_pretrained_file} "
                                     f"for model_path {model_path} but it's not available.")
        # Case 3: user didn't pass extension_pretrained_file, but pretrained_file attribute was
        # saved during training and is available.
        elif utils.is_url_or_existing_file(self._pretrained_file):
            extension_pretrained_file = self._pretrained_file
        # Case 4: no file is available, hope that pretrained embeddings weren't used in the first place and warn
        else:
            extra_info = (f"Originally pretrained_file was at "
                          f"{self._pretrained_file}. " if self._pretrained_file else "")
            # It's better to warn here and not give error because there is no way to distinguish between
            # whether pretrained-file wasn't used during training or user forgot to pass / passed incorrect
            # mapping. Raising an error would prevent fine-tuning in the former case.
            logging.warning(f"Embedding at model_path, {model_path} cannot locate the pretrained_file. "
                            f"{extra_info} If you are fine-tuning and want to use using pretrained_file for "
                            f"embedding extension, please pass the mapping by --embedding-sources argument.")

        embedding_dim = self.weight.data.shape[-1]
        if not extension_pretrained_file:
            extra_num_embeddings = extended_num_embeddings - self.num_embeddings
            extra_weight = torch.FloatTensor(extra_num_embeddings, embedding_dim)
            torch.nn.init.xavier_uniform_(extra_weight)
        else:
            # It's easiest to just reload the embeddings for the entire vocab,
            # then only keep the ones we need.
            whole_weight = utils._read_pretrained_embeddings_file(extension_pretrained_file, embedding_dim,
                                                            extended_vocab, vocab_namespace)
            extra_weight = whole_weight[self.num_embeddings:, :]

        extended_weight = torch.cat([self.weight.data, extra_weight], dim=0)
        self.weight = torch.nn.Parameter(extended_weight, requires_grad=self.weight.requires_grad)

    # Custom logic requires custom from_params.
    @classmethod
    def from_params(cls, vocab, params) -> 'Embedding':  # type: ignore
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
        key directly, and the vocabulary will be ignored.

        In the configuration file, a file containing pretrained embeddings can be specified
        using the parameter ``"pretrained_file"``.
        It can be the path to a local file or an URL of a (cached) remote file.
        Two formats are supported:

            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;

            * text file - an utf-8 encoded text file with space separated fields::

                    [word] [dim 1] [dim 2] ...

              The text file can eventually be compressed with gzip, bz2, lzma or zip.
              You can even select a single file inside an archive containing multiple files
              using the URI::

                    "(archive_uri)#file_path_inside_the_archive"

              where ``archive_uri`` can be a file system path or a URL. For example::

                    "(http://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"
        """
        # pylint: disable=arguments-differ
        num_embeddings = params.pop_int('num_embeddings', None)
        # If num_embeddings is present, set default namespace to None so that extend_vocab
        # call doesn't misinterpret that some namespace was originally used.
        vocab_namespace = params.pop("vocab_namespace", None if num_embeddings else "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop("pretrained_file", None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        params.assert_empty(cls.__name__)

        if pretrained_file:
            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.
            weight = utils._read_pretrained_embeddings_file(pretrained_file,
                                                      embedding_dim,
                                                      vocab,
                                                      vocab_namespace)
        else:
            weight = None

        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   weight=weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   vocab_namespace=vocab_namespace)



# ============== 2. from SolderedKG
class SpanWordAttention(torch.nn.Module):
    def __init__(self, config):
        super(SpanWordAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #self.query = get_linear_layer_init_identity(config.hidden_size)
        #self.key = get_linear_layer_init_identity(config.hidden_size)
        #self.value = get_linear_layer_init_identity(config.hidden_size)

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        """
        hidden_states = (batch_size, timesteps, dim)
        entity_embeddings = (batch_size, num_entities, dim)
        entity_mask = (batch_size, num_entities) with 0/1
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(entity_embeddings)
        mixed_value_layer = self.value(entity_embeddings)

        # (batch_size, num_heads, timesteps, head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # (batch_size, num_heads, num_entity_embeddings, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch_size, num_heads, timesteps, num_entity_embeddings)
        # gives the attention from timestep i to embedding j
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # apply the attention mask.
        # the attention_mask masks out thing to attend TO so we extend
        # the entity mask
        attention_mask = extend_attention_mask_for_bert(entity_mask, get_dtype_for_module(self))
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire entities to attend to, which might
        # seem a bit unusual, but is similar to the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # (batch_size, num_heads, timesteps, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, timesteps, num_heads, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # (batch_size, timesteps, hidden_dim)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs



class SpanAttention(torch.nn.Module):
    def __init__(self, config):
        super(SpanAttention, self).__init__()
        self.attention = SpanWordAttention(config)
        init_bert_weights(self.attention, config.initializer_range, (SpanWordAttention, ))
        self.output = BertSelfOutput(config)
        init_bert_weights(self.output, config.initializer_range)

    def forward(self, input_tensor, entity_embeddings, entity_mask):
        span_output, attention_probs = self.attention(input_tensor, entity_embeddings, entity_mask)
        attention_output = self.output(span_output, input_tensor)
        return attention_output, attention_probs



class SpanAttentionLayer(torch.nn.Module):
    # WARNING: does it's own init, so don't re-init
    def __init__(self, config):
        super(SpanAttentionLayer, self).__init__()
        self.attention = SpanAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        init_bert_weights(self.intermediate, config.initializer_range)
        init_bert_weights(self.output, config.initializer_range)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        attention_output, attention_probs = self.attention(hidden_states, entity_embeddings, entity_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return {"output": layer_output, "attention_probs": attention_probs}



class EntityLinkingWithCandidateMentions(EntityLinkingBase):
    def __init__(self,
                 entity_embedding,
                 contextual_embedding_dim: int = None,
                 span_encoder_config: Dict[str, int] = None,
                 margin: float = 0.2,
                 decode_threshold: float = 0.0,
                 loss_type: str = 'margin',
                 max_sequence_length: int = 512,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 include_null_embedding_in_dot_attention: bool = False,
                 namespace: str = 'entity',
                 ):
        super().__init__(
                         margin=margin,
                         decode_threshold=decode_threshold,
                         loss_type=loss_type,
                         namespace=namespace,
                         )

        entity_embedding_dim  = entity_embedding.output_dim

        if loss_type == 'margin':
            weighted_entity_threshold = decode_threshold
        else:
            weighted_entity_threshold = None

        null_entity_id = 470114
        unk_entity_id = 1

        self.disambiguator = EntityDisambiguator(
                 contextual_embedding_dim,
                 entity_embedding_dim=entity_embedding_dim,
                 entity_embeddings=entity_embedding,
                 max_sequence_length=max_sequence_length,
                 span_encoder_config=span_encoder_config,
                 dropout=dropout,
                 output_feed_forward_hidden_dim=output_feed_forward_hidden_dim,
                 initializer_range=initializer_range,
                 weighted_entity_threshold=weighted_entity_threshold,
                 include_null_embedding_in_dot_attention=include_null_embedding_in_dot_attention,
                 null_entity_id=null_entity_id)



    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset)
        return metrics


    def unfreeze(self, mode):
        # don't hold an parameters directly, so do nothing
        self.disambiguator.unfreeze(mode)

    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                gold_entities
                ):

        disambiguator_output = self.disambiguator(
            contextual_embeddings=contextual_embeddings,
            mask=tokens_mask,
            candidate_spans=candidate_spans,
            candidate_entities=candidate_entities['ids'],
            candidate_entity_priors=candidate_entity_priors,
            candidate_segment_ids=candidate_segment_ids,
            gold_entities=gold_entities
        )

        linking_scores = disambiguator_output['linking_scores']

        return_dict = disambiguator_output

        if gold_entities:
            loss_dict = self._compute_loss(
                    candidate_entities['ids'],
                    candidate_spans,
                    linking_scores,
                    gold_entities
            )
            return_dict.update(loss_dict)

        return return_dict



class WikiKG(nn.Module):
    def __init__(self,
                 entity_linker,
                 span_attention_config,
                 should_init_kg_to_bert_inverse: bool = True,
                 freeze: bool = False):
        super().__init__()
        # ====
        # self.contextual_embedding_dim = 768
        # self.dropout = 0.1
        # self.entity_embedding_dim = 300
        # self.entity_linker = EntityLinkingWithCandidateMentions()
        # self.kg_to_bert_projection = nn.Linear(300, 768)
        # 这里传递过来的是idx，不需要vocab了吧，在convert to id就该解决了
        # ====

        self.entity_linker = entity_linker
        self.entity_embedding_dim = self.entity_linker.disambiguator.entity_embedding_dim
        self.contextual_embedding_dim = self.entity_linker.disambiguator.contextual_embedding_dim

        self.should_init_kg_to_bert_inverse = should_init_kg_to_bert_inverse
        self._freeze_all = freeze

        self.weighted_entity_layer_norm = BertLayerNorm(self.entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.weighted_entity_layer_norm, 0.02)

        self.dropout = torch.nn.Dropout(0.1)

        # the span attention layers
        assert len(span_attention_config) == 4
        config = BertConfig(
            0, # vocab size, not used
            hidden_size=span_attention_config['hidden_size'],
            num_hidden_layers=span_attention_config['num_hidden_layers'],
            num_attention_heads=span_attention_config['num_attention_heads'],
            intermediate_size=span_attention_config['intermediate_size']
        )
        self.span_attention_layer = SpanAttentionLayer(config)
        # already init inside span attention layer

        # for the output!
        self.output_layer_norm = BertLayerNorm(self.contextual_embedding_dim, eps=1e-5)

        self.kg_to_bert_projection = torch.nn.Linear(
                self.entity_embedding_dim, self.contextual_embedding_dim)
        self._init_kg_to_bert_projection()



    def _init_kg_to_bert_projection(self):
        if not self.should_init_kg_to_bert_inverse:
            return

        # the output projection we initialize from the bert to kg, after
        # we load the weights
        # projection as the pseudo-inverse
        # w = (entity_dim, contextual_embedding_dim)
        w = self.entity_linker.disambiguator.bert_to_kg_projector.weight.data.detach().numpy()
        w_pseudo_inv = np.dot(np.linalg.inv(np.dot(w.T, w)), w.T)
        b = self.entity_linker.disambiguator.bert_to_kg_projector.bias.data.detach().numpy()
        b_pseudo_inv = np.dot(w_pseudo_inv, b)
        self.kg_to_bert_projection.weight.data.copy_(torch.tensor(w_pseudo_inv))
        self.kg_to_bert_projection.bias.data.copy_(torch.tensor(b_pseudo_inv))

    def get_metrics(self, reset=False):
        return self.entity_linker.get_metrics(reset)

    def unfreeze(self, mode):
        if self._freeze_all:
            for p in self.parameters():
                p.requires_grad_(False)
            self.entity_linker.unfreeze('freeze')
            return

        if mode == 'entity_linking':
            # training the entity linker, fix parameters here
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        # unfreeze will get called after loading weights in the case where
        # we pass a model archive to KnowBert, so re-init here
        self._init_kg_to_bert_projection()

        self.entity_linker.unfreeze(mode)

    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                gold_entities):

        linker_output = self.entity_linker(
                contextual_embeddings, tokens_mask,
                candidate_spans, candidate_entities, candidate_entity_priors,
                candidate_segment_ids, gold_entities)

        # update the span representations with the entity embeddings
        span_representations = linker_output['projected_span_representations']
        weighted_entity_embeddings = linker_output['weighted_entity_embeddings']
        spans_with_entities = self.weighted_entity_layer_norm(
                (span_representations +
                self.dropout(weighted_entity_embeddings)).contiguous()
        ) # 公式6

        # now run self attention between bert and spans_with_entities
        # to update bert.
        # this is done in projected dimension
        entity_mask = candidate_spans[:, :, 0] > -1
        span_attention_output = self.span_attention_layer(
                linker_output['projected_bert_representations'],
                spans_with_entities,
                entity_mask
        ) # 公式6和7之间的公式
        # entity enhanced
        projected_bert_representations_with_entities = span_attention_output['output']
        entity_attention_probs = span_attention_output["attention_probs"]

        # finally project back to full bert dimension!
        bert_representations_with_entities = self.kg_to_bert_projection(
                projected_bert_representations_with_entities
        )
        new_contextual_embeddings = self.output_layer_norm(
                (contextual_embeddings + self.dropout(bert_representations_with_entities)).contiguous()
        ) # 公式7

        return_dict = {'entity_attention_probs': entity_attention_probs,
                       'contextual_embeddings': new_contextual_embeddings,
                       'linking_scores': linker_output['linking_scores']}
        if 'loss' in linker_output:
            return_dict['loss'] = linker_output['loss']

        return return_dict



# ============== 2. from KnowBert

class BertPretrainedMetricsLoss(nn.Module):
    def __init__(self):
        super(BertPretrainedMetricsLoss, self).__init__()

        self.nsp_loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.lm_loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._metrics = {
            "total_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "nsp_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "lm_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "total_loss": Average(),
            "nsp_loss": Average(),
            "lm_loss": Average(),
            "lm_loss_wgt": WeightedAverage(),
            "mrr": MeanReciprocalRank(),
        }
        self._accuracy = CategoricalAccuracy()

    def get_metrics(self, reset: bool = False):
        metrics = {k: v.get_metric(reset) for k, v in self._metrics.items()}
        metrics['nsp_accuracy'] = self._accuracy.get_metric(reset)
        return metrics

    def _compute_loss(self,
                      contextual_embeddings,
                      pooled_output,
                      lm_label_ids,
                      next_sentence_label,
                      update_metrics=True):

        # (batch_size, timesteps, vocab_size), (batch_size, 2)
        prediction_scores, seq_relationship_score = self.pretraining_heads(
                contextual_embeddings, pooled_output
        )

        loss_metrics = []
        if lm_label_ids is not None:
            # Loss
            vocab_size = prediction_scores.shape[-1]
            masked_lm_loss = self.lm_loss_function(
                prediction_scores.view(-1, vocab_size), lm_label_ids["lm_labels"].view(-1)
            )
            masked_lm_loss_item = masked_lm_loss.item()
            loss_metrics.append([["lm_loss_ema", "lm_loss"], masked_lm_loss_item])
            num_lm_predictions = (lm_label_ids["lm_labels"] > 0).long().sum().item()
            self._metrics['lm_loss_wgt'](masked_lm_loss_item, num_lm_predictions)
        else:
            masked_lm_loss = 0.0

        if next_sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            loss_metrics.append([["nsp_loss_ema", "nsp_loss"], next_sentence_loss.item()])
            if update_metrics:
                self._accuracy(
                    seq_relationship_score.detach(), next_sentence_label.view(-1).detach()
                )
        else:
            next_sentence_loss = 0.0

        # update metrics
        if update_metrics:
            total_loss = masked_lm_loss + next_sentence_loss
            for keys, v in [[["total_loss_ema", "total_loss"], total_loss.item()]] + loss_metrics:
                for key in keys:
                    self._metrics[key](v)

        return masked_lm_loss, next_sentence_loss

    def _compute_mrr(self,
                     contextual_embeddings,
                     pooled_output,
                     lm_labels_ids,
                     mask_indicator):
        prediction_scores, seq_relationship_score = self.pretraining_heads(
                contextual_embeddings, pooled_output
        )
        self._metrics['mrr'](prediction_scores, lm_labels_ids, mask_indicator)



class KnowBert(BertPretrainedMetricsLoss):
    """
        KnowBert: Combines bert with one or more SolderedKG
        each SolderedKG is inserted at a particular level, given by an index,
        such that we run Bert to the index, then the SolderedKG, then the rest
        of bert.  Indices such that index 0 means run the first contextual layer,
        then add KG, and index 11 means run to the top of Bert, then the KG
        (for bert base with 12 layers).

    """
    def __init__(self,
                 soldered_kgs,
                 soldered_layers: Dict[str, int],
                 bert_model_name: str,
                 mode: str = None,
                 model_archive: str = None,
                 strict_load_archive: bool = True,
                 debug_cuda: bool = False,
                 remap_segment_embeddings: int = None,
                 ):

        super().__init__()

        self.remap_segment_embeddings = remap_segment_embeddings

        # get the LM + NSP parameters from BERT
        pretrained_bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.pretrained_bert = pretrained_bert
        self.pretraining_heads = pretrained_bert.cls
        self.pooler = pretrained_bert.bert.pooler

        # the soldered kgs
        self.soldered_kgs = soldered_kgs
        for key, skg in soldered_kgs.items():
            self.add_module(key + "_soldered_kg", skg)

        # list of (layer_number, soldered key) sorted in ascending order
        self.layer_to_soldered_kg = sorted(
                [(layer, key) for key, layer in soldered_layers.items()]
        )
        # the last layer
        num_bert_layers = len(self.pretrained_bert.bert.encoder.layer)
        # the first element of the list is the index
        self.layer_to_soldered_kg.append([num_bert_layers - 1, None])

        if model_archive is not None:
            with tarfile.open(cached_path(model_archive), 'r:gz') as fin:
                # a file object
                weights_file = fin.extractfile('weights.th')
                state_dict = torch.load(weights_file, map_location=utils.device_mapping(-1))
            self.load_state_dict(state_dict, strict=strict_load_archive)

        if remap_segment_embeddings is not None:
            # will redefine the segment embeddings
            new_embeddings = self._remap_embeddings(self.pretrained_bert.bert.embeddings.token_type_embeddings.weight)
            if new_embeddings is not None:
                del self.pretrained_bert.bert.embeddings.token_type_embeddings
                self.pretrained_bert.bert.embeddings.token_type_embeddings = new_embeddings

        assert mode in (None, 'entity_linking')
        self.mode = mode
        self.unfreeze()

        if debug_cuda:
            for m in self.modules():
                m.register_forward_hook(diagnose_forward_hook)
                m.register_backward_hook(diagnose_backward_hook)

    def _remap_embeddings(self, token_type_embeddings):
        embed_dim = token_type_embeddings.shape[1]
        if list(token_type_embeddings.shape) == [self.remap_segment_embeddings, embed_dim]:
            # already remapped!
            return None
        new_embeddings = torch.nn.Embedding(self.remap_segment_embeddings, embed_dim)
        new_embeddings.weight.data.copy_(token_type_embeddings.data[0, :])
        return new_embeddings


    def load_state_dict(self, state_dict, strict=True):
        if self.remap_segment_embeddings:
            # hack the embeddings!
            new_embeddings = self._remap_embeddings(state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'])
            if new_embeddings is not None:
                state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'] = new_embeddings.weight
        super().load_state_dict(state_dict, strict=strict)

    def unfreeze(self):
        if self.mode == 'entity_linking':
            # all parameters in BERT are fixed, just training the linker
            # linker specific params set below when calling soldered_kg.unfreeze
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module.unfreeze(self.mode)

    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module_metrics = module.get_metrics(reset)
            for metric_key, val in module_metrics.items():
                metrics[key + '_' + metric_key] = val

        return metrics


    def forward(self, tokens=None, segment_ids=None, candidates=None,
                candidate_spans=None, candidate_entities=None, candidate_entity_priors=None, candidate_segment_ids=None,
                lm_label_ids=None, next_sentence_label=None, gold_entities=None):
        # tokens & segment_ids: 16*169, gold_entities: {wiki: 16*47*1}
        assert candidates.keys() == self.soldered_kgs.keys() # wiki

        mask = tokens > 0
        attention_mask = extend_attention_mask_for_bert(mask, get_dtype_for_module(self)) # 16, 1, 1, 109
        contextual_embeddings = self.pretrained_bert.bert.embeddings(tokens, segment_ids.to(torch.long)) # 16*109*768

        output = {}
        start_layer_index = 0
        loss = 0.0

        for layer_num, soldered_kg_key in self.layer_to_soldered_kg:
            end_layer_index = layer_num + 1
            if end_layer_index > start_layer_index:
                # run bert from start to end layers
                # 这种实现方式更好，不用更改模型框架了呀！！！！！！！！！！！！！！！
                layer_list = self.pretrained_bert.bert.encoder.layer[start_layer_index:end_layer_index]
                for layer in layer_list:
                    contextual_embeddings = layer(contextual_embeddings, attention_mask)
            start_layer_index = end_layer_index

            # run the SolderedKG component
            if soldered_kg_key is not None:
                soldered_kg = getattr(self, soldered_kg_key + "_soldered_kg")
                # soldered_kwargs = candidates[soldered_kg_key]
                # if gold_entities is not None and soldered_kg_key in gold_entities:
                #     soldered_kwargs['gold_entities'] = gold_entities[soldered_kg_key]
                kg_output = soldered_kg(
                        contextual_embeddings,
                        mask,
                        candidate_spans, candidate_entities, candidate_entity_priors,
                        candidate_segment_ids, gold_entities)

                if 'loss' in kg_output:
                    loss = loss + kg_output['loss']

                contextual_embeddings = kg_output['contextual_embeddings']
                output[soldered_kg_key] = {}
                for key in kg_output.keys():
                    if key != 'loss' and key != 'contextual_embeddings':
                        output[soldered_kg_key][key] = kg_output[key]

        # get the pooled CLS output
        pooled_output = self.pooler(contextual_embeddings)

        output['loss'] = loss
        output['pooled_output'] = pooled_output
        output['contextual_embeddings'] = contextual_embeddings

        return output


if __name__ == "__main__":
    # If we're loading a saved model, we don't want to actually read a pre-trained
    # embedding file - the embeddings will just be in our saved weights, and we might not
    # have the original embedding file anymore, anyway.
    entity_vocab_file = 'G:/D/phd4/XEmbedding/data/wiki_linker/vocabulary_wiki/entity.txt'
    entity_embeddings_file = 'G:/D/phd4/XEmbedding/data/wiki_linker/entities_glove_format/entities_glove_format'

    from data_utils import WikiEntityLoader
    wikientity_loader = WikiEntityLoader(entity_vocab_file, entity_embeddings_file)
    weight = wikientity_loader.embedding_matrix

    # trainable=False ??
    entity_embedding = Embedding(num_embeddings=470116, embedding_dim=300, weight=weight, trainable=False, vocab_namespace='entity')

    # span_encoder_config: Dict[str, int] = None,
    span_encoder_config = {'hidden_size': 300, 'intermediate_size': 1024, 'num_attention_heads': 4, 'num_hidden_layers': 1}

    entity_linker = EntityLinkingWithCandidateMentions(entity_embedding=entity_embedding,
                                                       contextual_embedding_dim=768,
                                                       span_encoder_config=span_encoder_config,
                                                       )

    wiki_kg = WikiKG(entity_linker=entity_linker, span_attention_config=span_encoder_config)

    soldered_kgs = {"wiki": wiki_kg}
    soldered_layers = {"wiki": 9}

    knowbert = KnowBert(soldered_kgs=soldered_kgs,
                        soldered_layers=soldered_layers,
                        bert_model_name='bert-base-uncased', mode='entity_linking')

    t = knowbert