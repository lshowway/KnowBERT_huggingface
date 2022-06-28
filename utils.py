import os
import torch
import numpy
import h5py
from tqdm import tqdm


import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


from typing import Dict, MutableMapping, Mapping, Optional, Union, List
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField, ListField, SpanField

from urllib.parse import urlparse
from pathlib import Path

from pytorch_pretrained_bert.modeling import \
        BertLayer, BertAttention, BertSelfAttention, BertSelfOutput, \
        BertOutput, BertIntermediate, BertEncoder, BertLayerNorm, BertConfig



def remap_span_indices_after_subword_tokenization(
        original_span_indices, offsets, num_wordpieces
    ):
    """
    ASSUMED original data has CLS/SEP and then was wordpiece tokenized
    use_starting_offsets = True for the indexer and the offsets!
    num_wordpieces = number of total word pieces, including CLS and SEP
    """
    end_offsets = [e - 1 for e in offsets[1:]] + [num_wordpieces - 2]
    new_span_indices = [
        [offsets[s], end_offsets[e]] for s, e in original_span_indices
    ]
    return new_span_indices



class Instance(Mapping[str, Field]):
    """
    An ``Instance`` is a collection of :class:`~allennlp.data.fields.field.Field` objects,
    specifying the inputs and outputs to
    some model.  We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.

    The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will be indexed, after which multiple instances can be combined
    into a ``Batch`` and then converted into padded arrays.

    Parameters
    ----------
    fields : ``Dict[str, Field]``
        The ``Field`` objects that will be used to produce data arrays for this instance.
    """
    def __init__(self, fields: MutableMapping[str, Field],
                       should_remap_span_indices: bool = False) -> None:
        self.fields = fields
        self.indexed = False
        self.should_remap_span_indices = should_remap_span_indices

    # Add methods for ``Mapping``.  Note, even though the fields are
    # mutable, we don't implement ``MutableMapping`` because we want
    # you to use ``add_field`` and supply a vocabulary.
    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def add_field(self, field_name: str, field: Field, vocab: Vocabulary = None) -> None:
        """
        Add the field to the existing fields mapping.
        If we have already indexed the Instance, then we also index `field`, so
        it is necessary to supply the vocab.
        """
        self.fields[field_name] = field
        if self.indexed:
            field.index(vocab)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given ``counter`` for all of the vocabulary items in all of the
        ``Fields`` in this ``Instance``.
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) -> None:
        """
        Indexes all fields in this ``Instance`` using the provided ``Vocabulary``.
        This `mutates` the current object, it does not return a new ``Instance``.
        A ``DataIterator`` will call this on each pass through a dataset; we use the ``indexed``
        flag to make sure that indexing only happens once.

        This means that if for some reason you modify your vocabulary after you've
        indexed your instances, you might get unexpected behavior.
        """
        if not self.indexed:
            self.indexed = True
            offsets = None

            for field in self.fields.values():
                field.index(vocab)

                if self.should_remap_span_indices:
                    if isinstance(field, TextField):
                        for key in field._indexed_tokens:
                            if key[-8:] == '-offsets':
                                if offsets is not None:
                                    raise ValueError
                                offsets = field._indexed_tokens[key]
                                key_no_offsets = key[:-8]
                                num_wordpieces = len(field._indexed_tokens[key_no_offsets])

            if offsets is not None:
                # some wordpiece indexer -- look for span fields to modify
                for field in self.fields.values():
                    if isinstance(field, ListField) and isinstance(field.field_list[0], SpanField):
                        # need to modify the span fields
                        old_spans = [
                            [span.span_start, span.span_end] for span in field.field_list
                        ]
                        new_spans = remap_span_indices_after_subword_tokenization(
                            old_spans, offsets, num_wordpieces
                        )
                        for span, (new_start, new_end) in zip(field.field_list, new_spans):
                            span.span_start = new_start
                            span.span_end = new_end

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed by field name.  Each ``Field`` returns a
        mapping from padding keys to actual lengths, and we just key that dictionary by field name.
        """
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_tensor_dict(self,
                       padding_lengths: Dict[str, Dict[str, int]] = None) -> Dict[str, DataArray]:
        """
        Pads each ``Field`` in this instance to the lengths given in ``padding_lengths`` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of torch tensors for each field.

        If ``padding_lengths`` is omitted, we will call ``self.get_padding_lengths()`` to get the
        sizes of the tensors to create.
        """
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors


    def __str__(self) -> str:
        base_string = f"Instance with fields:\n"
        return " ".join([base_string] + [f"\t {name}: {field} \n"
                                         for name, field in self.fields.items()])



# @Metric.register("f1_set")
class F1Metric():
    """
    A generic set based F1 metric.
    Takes two lists of predicted and gold elements and computes F1.
    Only requirements are that the elements are hashable.
    """
    def __init__(self, filter_func=None):
        self.reset()
        if filter_func is None:
            filter_func = lambda x: True
        self.filter_func = filter_func

    def reset(self):
        self._true_positives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def __call__(self, predictions, gold_labels):
        """
        predictions = batch of predictions that can be compared
        gold labels = list of gold labels

        e.g.
            predictions = [
                 [('ORG', (0, 1)), ('PER', (5, 8))],
                 [('MISC', (9, 13))]
            ]
            gold_labels = [
                [('ORG', (0, 1))],
                []
            ]

        elements must be hashable
        """
        assert len(predictions) == len(gold_labels)

        for pred, gold in zip(predictions, gold_labels):
            s_gold = set(g for g in gold if self.filter_func(g))
            s_pred = set(p for p in pred if self.filter_func(p))

            for p in s_pred:
                if p in s_gold:
                    self._true_positives += 1
                else:
                    self._false_positives += 1

            for p in s_gold:
                if p not in s_pred:
                    self._false_negatives += 1


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length)
    Return a view that's (d1 * ... * dn, sequence_length).
    If original tensor is 1-d or 2-d, return it as is.
    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> torch.Tensor:
    """
    Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape
    (d1, ..., dn, sequence_length),
    return the reshaped tensor of embeddings with shape
    (d1, ..., dn, sequence_length, embedding_dim).
    If original size is 1-d or 2-d, return it as is.
    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)



class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional ``pass_through`` iterable.
    """
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):
        # pylint: disable=arguments-differ
        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)


def init_bert_weights(module, initializer_range, extra_modules_without_weights=()):
    # these modules don't have any weights, other then ones in submodules,
    # so don't have to worry about init
    modules_without_weights = (
        BertEncoder, torch.nn.ModuleList, torch.nn.Dropout, BertLayer,
        BertAttention, BertSelfAttention, BertSelfOutput,
        BertOutput, BertIntermediate
    ) + extra_modules_without_weights


    # modified from pytorch_pretrained_bert
    def _do_init(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            m.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(m, BertLayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        elif isinstance(m, modules_without_weights):
            pass
        else:
            raise ValueError(str(m))

        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    for mm in module.modules():
        _do_init(mm)


def extend_attention_mask_for_bert(mask, dtype):
    # mask = (batch_size, timesteps)
    # returns an attention_mask useable with BERT
    # see: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L696
    extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_dtype_for_module(module):
    # gets dtype for module parameters, for fp16 support when casting
    # we unfortunately can't set this during module construction as module
    # will be moved to GPU or cast to half after construction.
    return next(module.parameters()).dtype


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)



def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()



def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for :func:`~batched_index_select`. The given ``indices`` of size
    ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor, which has size
    ``(batch_size, sequence_length, embedding_size)``. This function returns a vector that
    correctly indexes into the flattened target. The sequence length of the target must be
    provided to compute the appropriate offsets.

    .. code-block:: python

        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]

    Parameters
    ----------
    indices : ``torch.LongTensor``, required.
    sequence_length : ``int``, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    Returns
    -------
    offset_indices : ``torch.LongTensor``
    """
    # Shape: (batch_size)
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """
    The given ``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into the sequence
    dimension (dimension 2) of the target, which has size ``(batch_size, sequence_length,
    embedding_size)``.

    This function returns selected values in the target with respect to the provided indices, which
    have size ``(batch_size, d_1, ..., d_n, embedding_size)``. This can use the optionally
    precomputed :func:`~flattened_indices` with size ``(batch_size * d_1 * ... * d_n)`` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    :class:`~allennlp.models.coreference_resolution.CoreferenceResolver`. Model to select
    contextual word representations corresponding to the start and end indices of mentions. The key
    reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : ``torch.LongTensor``
        A tensor of shape (batch_size, ...), where each element is an index into the
        ``sequence_length`` dimension of the ``target`` tensor.
    flattened_indices : Optional[torch.Tensor], optional (default = None)
        An optional tensor representing the result of calling :func:~`flatten_and_batch_shift_indices`
        on ``indices``. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    Returns
    -------
    selected_targets : ``torch.Tensor``
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets



def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        dtype = vector.dtype
        mask = mask.to(dtype)
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool:
    """
    Given something that might be a URL (or might be a local path),
    determine check if it's url or an existing file path.
    """
    if url_or_filename is None:
        return False
    url_or_filename = os.path.expanduser(str(url_or_filename))
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https', 's3') or os.path.exists(url_or_filename)





def _read_embeddings_from_hdf5(embeddings_filename: str,
                               embedding_dim: int,
                               vocab: Vocabulary,
                               namespace: str = "tokens") -> torch.FloatTensor:
    """
    Reads from a hdf5 formatted file. The embedding matrix is assumed to
    be keyed by 'embedding' and of size ``(num_tokens, embedding_dim)``.
    """
    with h5py.File(embeddings_filename, 'r') as fin:
        embeddings = fin['embedding'][...]

    if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
        raise Exception(
                "Read shape {0} embeddings from the file, but expected {1}".format(
                        list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]))

    return torch.FloatTensor(embeddings)



def get_file_extension(path: str, dot=True, lower: bool = True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext






