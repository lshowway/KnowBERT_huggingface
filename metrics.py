import numpy as np
from overrides import overrides
from allennlp.training.metrics.metric import Metric
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import torch
from typing import Optional



class ExponentialMovingAverage(Metric):
    """
    Keep an exponentially weighted moving average.
    alpha is the decay constant. Alpha = 1 means just keep the most recent value.
    alpha = 0.5 will have almost no contribution from 10 time steps ago.
    """
    def __init__(self, alpha:float = 0.5) -> None:
        self.alpha = alpha
        self.reset()

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        if self._ema is None:
            # first observation
            self._ema = value
        else:
            self._ema = self.alpha * value + (1.0 - self.alpha) * self._ema

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        if self._ema is None:
            ret = 0.0
        else:
            ret = self._ema

        if reset:
            self.reset()

        return ret

    @overrides
    def reset(self):
        self._ema = None


class Average(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value += list(self.unwrap_to_tensors(value))[0]
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0



class WeightedAverage(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value, count=1):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value += (list(self.unwrap_to_tensors(value))[0] * count)
        self._count += count

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0


class MeanReciprocalRank(Metric):
    def __init__(self):
        self._sum = 0.0
        self._n = 0.0

    def __call__(self, predictions, labels, mask):
        # Flatten
        labels = labels.view(-1)
        mask = mask.view(-1).float()
        predictions = predictions.view(labels.shape[0], -1)

        # MRR computation
        label_scores = predictions.gather(-1, labels.unsqueeze(-1))
        rank = predictions.ge(label_scores).sum(1).float()
        reciprocal_rank = 1 / rank
        self._sum += (reciprocal_rank * mask).sum().item()
        self._n += mask.sum().item()

    def get_metric(self, reset=False):
        mrr = self._sum / (self._n + 1e-13)
        if reset:
            self.reset()
        return mrr

    @overrides
    def reset(self):
        self._sum = 0.0
        self._n = 0.0



class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """
    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise Exception("Tie break in Categorical Accuracy "
                                     "can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise Exception("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise Exception("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise Exception("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[torch.arange(gold_labels.numel()).long(), gold_labels].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0



