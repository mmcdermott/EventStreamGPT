import abc

import torch

from ..data.types import PytorchBatch
from .config import StructuredTransformerConfig


class Labeler(abc.ABC):
    """A base class for zero-shot labeler functors.

    Zero-shot labeler functors are used to enable users to run zero-shot evaluation over novel fine-tuning
    tasks. To produce a zero-shot labeler, users must:

    1. Sub-class this base class in a new file.
    2. Implement the `__call__` method; this method must take as input a batch object `batch`, which will
       contain newly generated data, and an integral `input_seq_len` parameter which gives how long the input
       sequence was prior to generation. It must return a tuple of tensors -- first, a `torch.LongTensor`
       containing one-hot classification labels that are implied by the generated sequences in the batch
       elements, and second, a `torch.BoolTensor` which indicates for each element of the generated set of
       labels whether or not a label was able to be produced for that sample.
    3. Copy the file containing this labeler class into the task directory with the name
       `${task_df_name}_labeler.py`.

    You can then use built-in zero-shot evaluation utilities on that task and your labeler will automatically
    be used to evaluate zero-shot performance via unsupervised generation.

    Attributes:
        config: The `StructuredTransformerConfig` config object defining the model being used. This holds
            information about vocabulary elements, index maps (which is important to decipher batch data into
            categories), etc.

    .. automethod:: __call__
    """

    def __init__(self, config: StructuredTransformerConfig):
        self.config = config

    @abc.abstractmethod
    def __call__(self, batch: PytorchBatch, input_seq_len: int) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """The core labeling method of the class. Must be overwritten by subclass.

        Args:
            batch: The PyTorch Batch, containing both the initial raw input data (left padded), followed by
                the newly generated data.
            input_seq_len: The number of events (including padding) on the left side of the batch that were
                the original raw input, rather than the newly generated data. E.g., `batch[: :input_seq_len]`
                is just events that were in the original input, and `batch[:, input_seq_len:]` is just the
                newly generated events.

        Returns:
            torch.LongTensor: The classification labels (in one-hot, [batch_size x vocab_size] format) that
                the labeler has generated in response to the input batch.
            torch.BoolTensor: A boolean tensor of shape [batch_size] indicating whether or not each sample in
                the original input were able to be parsed into a label (`True`) or not (`False`).
        """
        raise NotImplementedError("Must be overwritten by a subclass!")
