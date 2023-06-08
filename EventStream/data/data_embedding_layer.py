import enum
from typing import Union

import torch

from ..utils import StrEnum
from .types import PytorchBatch


class EmbeddingMode(StrEnum):
    """The different ways that the data can be embedded."""

    JOINT = enum.auto()
    """Embed all data jointly via a single embedding layer, weighting observed measurement
    embdddings by values when present."""

    SPLIT_CATEGORICAL_NUMERICAL = enum.auto()
    """Embed the categorical observations of measurements separately from their numerical values,
    and combine the two via a specifiable strategy."""


class MeasIndexGroupOptions(StrEnum):
    """The different ways that the `split_by_measurement_indices` argument can be interpreted.

    If measurements are split, then the final embedding can be seen as a combination of
    ``emb_cat(measurement_indices)`` and ``emb_num(measurement_indices, measurement_values)``, where
    ``emb_*`` are embedding layers with sum aggregations that take in indices to be embedded and
    possible values to use in the output sum. This enumeration controls how those two elements are
    combined for a given measurement feature.
    """

    CATEGORICAL_ONLY = enum.auto()
    """Only embed the categorical component of this measurement (``emb_cat(...)``)."""

    CATEGORICAL_AND_NUMERICAL = enum.auto()
    """Embed both the categorical features and the numerical features of this measurement."""

    NUMERICAL_ONLY = enum.auto()
    """Only embed the numerical component of this measurement (``emb_num(...)``)."""


MEAS_INDEX_GROUP_T = Union[int, tuple[int, MeasIndexGroupOptions]]


class StaticEmbeddingMode(StrEnum):
    """The different ways that static embeddings can be combined with the dynamic embeddings."""

    DROP = enum.auto()
    """Static embeddings are dropped, and only the dynamic embeddings are used."""

    SUM_ALL = enum.auto()
    """Static embeddings are summed with the dynamic embeddings per event."""


class DataEmbeddingLayer(torch.nn.Module):
    """This class efficiently embeds an PytorchDataset batch's metadata into a fixed-size embedding
    layer, taking into account `dynamic_indices` (including an implicit padding index of 0),
    `dynamic_values`, and `dynamic_values_mask`.

    It *does not* take into account `data_types`, `time`, or `event_mask`.
    """

    def __init__(
        self,
        n_total_embeddings: int,
        out_dim: int,
        static_embedding_mode: StaticEmbeddingMode,
        categorical_embedding_dim: int | None = None,
        numerical_embedding_dim: int | None = None,
        split_by_measurement_indices: list[list[MEAS_INDEX_GROUP_T]] | None = None,
        do_normalize_by_measurement_index: bool = False,
        static_weight: float = 1 / 2,
        dynamic_weight: float = 1 / 2,
        categorical_weight: float = 1 / 2,
        numerical_weight: float = 1 / 2,
    ):
        """Initializes the DataEmbeddingLayer. Uses a padding index of 0, as that is how the
        `PytorchDataset` object is structured.

        Args:
            `n_total_embeddings` (`int`): The total vocabulary size that needs to be embedded.
            `out_dim` (`int`): The output dimension of the embedding layer.
            `static_embedding_mode` (`StaticEmbeddingMode`):
                The way that static embeddings are combined with the dynamic embeddings.
            `categorical_embedding_dim` (`Optional[int]`, *default* = `None`):
                The dimension of the categorical embeddings. If `None`, no separate categorical embeddings
                are used.
            `numerical_embedding_dim` (`Optional[int]`, *default* = `None`):
                The dimension of the numerical embeddings. If `None`, no separate numerical embeddings
                are used.
            `split_by_measurement_indices` (`Optional[List[List[MEAS_INDEX_GROUP_T]]]`, *default* = `None`):
                If not `None`, then the `dynamic_indices` are split into multiple groups, and each group is
                embedded separately. The `split_by_measurement_indices` argument is a list of lists of
                indices. Each inner list is a group of indices that will be embedded separately. Each index
                can be an integer, in which case it is the index of the measurement to be embedded, or it
                can be a tuple of the form `(index, embed_only_categorical)`, in which case `index` is the
                index of the measurement to be embedded, and `embed_only_categorical` indicates
                whether the group includes only the categorical index of the measurement, only the numerical
                value of the measurement, or both its categorical index and it's numerical values.

                Note that measurement index groups are assumed to only apply to the dynamic indices, not the
                static indices, as static indices are never generated and should be assumed to be causally
                linked to all elements of a given event.
            `do_normalize_by_measurement_index` (`bool`, *default* = `False`):
                If `True`, then the embeddings of each measurement are normalized by the number of
                measurements of that `measurement_index` in the batch.
            `static_weight` (`float`, *default* = `1/2`):
                The weight of the static embeddings. Only used if `static_embedding_mode` is not
                `StaticEmbeddingMode.DROP`.
            `dynamic_weight` (`float`, *default* = `1/2`):
                The weight of the dynamic embeddings. Only used if `static_embedding_mode` is not
                `StaticEmbeddingMode.DROP`.
            `categorical_weight` (`float`, *default* = `1/2`):
                The weight of the categorical embeddings. Only used if `categorical_embedding_dim` and
                `numerical_embedding_dim` are not `None`.
            `numerical_weight` (`float`, *default* = `1/2`):
                The weight of the numerical embeddings. Only used if `categorical_embedding_dim` and
                `numerical_embedding_dim` are not `None`.
        """
        super().__init__()

        if type(out_dim) is not int:
            raise TypeError("`out_dim` must be an `int`.")
        if out_dim <= 0:
            raise ValueError("`out_dim` must be positive.")
        if type(n_total_embeddings) is not int:
            raise TypeError("`n_total_embeddings` must be an `int`.")
        if n_total_embeddings <= 0:
            raise ValueError("`n_total_embeddings` must be positive.")

        if static_embedding_mode not in StaticEmbeddingMode.values():
            raise TypeError(
                "`static_embedding_mode` must be a `StaticEmbeddingMode` enum member: "
                f"{StaticEmbeddingMode.values()}."
            )
        if (categorical_embedding_dim is not None) or (numerical_embedding_dim is not None):
            if (categorical_embedding_dim is None) or (numerical_embedding_dim is None):
                raise ValueError(
                    "If either `categorical_embedding_dim` or `numerical_embedding_dim` is not `None`, "
                    "then both must be not `None`."
                )
            if type(categorical_embedding_dim) is not int:
                raise TypeError("`categorical_embedding_dim` must be an `int`.")
            if categorical_embedding_dim <= 0:
                raise ValueError("`categorical_embedding_dim` must be positive.")
            if type(numerical_embedding_dim) is not int:
                raise TypeError("`numerical_embedding_dim` must be an `int`.")
            if numerical_embedding_dim <= 0:
                raise ValueError("`numerical_embedding_dim` must be positive.")

        if split_by_measurement_indices is not None:
            for group in split_by_measurement_indices:
                if type(group) is not list:
                    raise TypeError("`split_by_measurement_indices` must be a list of lists.")
                for index in group:
                    if not isinstance(index, (int, tuple)):
                        raise TypeError(
                            "`split_by_measurement_indices` must be a list of lists of ints and/or tuples."
                        )
                    if type(index) is tuple:
                        if len(index) != 2:
                            raise ValueError(
                                "Each tuple in `split_by_measurement_indices` must have length 2."
                            )
                        index, embed_mode = index
                        if type(index) is not int:
                            raise TypeError(
                                "The first element of each tuple in each list of "
                                "`split_by_measurement_indices` must be an int."
                            )
                        if embed_mode not in MeasIndexGroupOptions.values():
                            raise TypeError(
                                "The second element of each tuple in each sublist of "
                                "`split_by_measurement_indices` must be a member of the "
                                f"`MeasIndexGroupOptions` enum: {MeasIndexGroupOptions.values()}."
                            )

        self.out_dim = out_dim
        self.static_embedding_mode = static_embedding_mode
        self.split_by_measurement_indices = split_by_measurement_indices
        self.do_normalize_by_measurement_index = do_normalize_by_measurement_index
        self.static_weight = static_weight / (static_weight + dynamic_weight)
        self.dynamic_weight = dynamic_weight / (static_weight + dynamic_weight)
        self.categorical_weight = categorical_weight / (categorical_weight + numerical_weight)
        self.numerical_weight = numerical_weight / (categorical_weight + numerical_weight)

        self.n_total_embeddings = n_total_embeddings

        if categorical_embedding_dim is None and numerical_embedding_dim is None:
            self.embedding_mode = EmbeddingMode.JOINT
            self.embed_layer = torch.nn.EmbeddingBag(
                num_embeddings=n_total_embeddings,
                embedding_dim=out_dim,
                mode="sum",
                padding_idx=0,
            )
        else:
            self.embedding_mode = EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL
            assert (categorical_embedding_dim is not None) and (numerical_embedding_dim is not None)
            self.categorical_embed_layer = torch.nn.EmbeddingBag(
                num_embeddings=n_total_embeddings,
                embedding_dim=categorical_embedding_dim,
                mode="sum",
                padding_idx=0,
            )
            self.cat_proj = torch.nn.Linear(categorical_embedding_dim, out_dim)
            self.numerical_embed_layer = torch.nn.EmbeddingBag(
                num_embeddings=n_total_embeddings,
                embedding_dim=numerical_embedding_dim,
                mode="sum",
                padding_idx=0,
            )
            self.num_proj = torch.nn.Linear(numerical_embedding_dim, out_dim)

    @staticmethod
    def get_measurement_index_normalziation(measurement_indices: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of shape `(batch_size, num_measurements)` that can be used to normalize
        the embeddings of each measurement by the number of measurements in the batch.

        Args:
            `measurement_indices` (`torch.Tensor`):
                A tensor of shape `(batch_size, num_measurements)` that contains the indices of the
                measurements in each batch element. Zero indicates padded measurements
                and the returned mask will have a value of zero in those positions.

        Returns:
            `torch.Tensor`:
                A tensor of shape `(batch_size, num_measurements)` that can be used to normalize the
                embeddings of each measurement by the number of measurements in the batch.
        """

        one_hot = torch.nn.functional.one_hot(measurement_indices)

        normalization_vals = 1.0 / one_hot.sum(dim=-2)

        normalization_vals = torch.gather(normalization_vals, dim=-1, index=measurement_indices)

        # Make sure that the zero index is not counted in the normalization
        normalization_vals = torch.where(measurement_indices == 0, 0, normalization_vals)

        normalization_vals_sum = normalization_vals.sum(dim=-1, keepdim=True)
        normalization_vals_sum = torch.where(normalization_vals_sum == 0, 1, normalization_vals_sum)
        return normalization_vals / normalization_vals_sum

    def joint_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values is None:
            values = torch.ones_like(indices, dtype=torch.float32)
        else:
            values = torch.where(values_mask, values, 1)

        if self.do_normalize_by_measurement_index:
            values *= self.get_measurement_index_normalziation(measurement_indices)

        return self.embed_layer(indices, per_sample_weights=values)

    def split_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
        cat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cat_values = torch.ones_like(indices, dtype=torch.float32)
        if cat_mask is not None:
            cat_values = torch.where(cat_mask, cat_values, 0)
        if self.do_normalize_by_measurement_index:
            meas_norm = self.get_measurement_index_normalziation(measurement_indices)
            cat_values *= meas_norm

        cat_embeds = self.cat_proj(self.categorical_embed_layer(indices, per_sample_weights=cat_values))

        if values is None:
            return cat_embeds

        num_values = torch.where(values_mask, values, 0)
        if self.do_normalize_by_measurement_index:
            num_values *= meas_norm

        num_embeds = self.num_proj(self.numerical_embed_layer(indices, per_sample_weights=num_values))

        return self.categorical_weight * cat_embeds + self.numerical_weight * num_embeds

    def embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
        cat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        torch._assert(
            indices.max() < self.n_total_embeddings,
            f"Invalid embedding! {indices.max()} >= {self.n_total_embeddings}",
        )
        match self.embedding_mode:
            case EmbeddingMode.JOINT:
                return self.joint_embed(indices, measurement_indices, values, values_mask)
            case EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL:
                return self.split_embed(indices, measurement_indices, values, values_mask, cat_mask)
            case _:
                raise ValueError(f"Invalid embedding mode: {self.embedding_mode}")

    def static_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        return self.embed(batch["static_indices"], batch["static_measurement_indices"])

    def split_batch_into_measurement_index_buckets(
        self, batch: PytorchBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the batch into groups of measurement indices, and returns the indices, values, and
        measurement indices for each bucket.

        Args:
            `batch` (`PytorchBatch`): A batch of data.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                A tuple of tensors that contain the categorical mask and values mask for each group.
        """
        batch_size, sequence_length, num_data_elements = batch["dynamic_measurement_indices"].shape

        categorical_masks = []
        numerical_masks = []
        for i, meas_index_group in enumerate(self.split_by_measurement_indices):
            if len(meas_index_group) == 0 and i > 0:
                raise ValueError(
                    f"Empty measurement index group: {meas_index_group} at index {i}! "
                    "Only the first (i=0) group can be empty (in cases where there are no "
                    "FUNCTIONAL_TIME_DEPENDENT measurements)."
                )

            # Create a mask that is True if each data element in the batch is in the measurement group.
            group_categorical_mask = torch.zeros_like(batch["dynamic_measurement_indices"], dtype=torch.bool)
            group_values_mask = torch.zeros_like(batch["dynamic_measurement_indices"], dtype=torch.bool)
            for meas_index in meas_index_group:
                if type(meas_index) is tuple:
                    meas_index, group_mode = meas_index
                else:
                    group_mode = MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL

                new_index_mask = batch["dynamic_measurement_indices"] == meas_index
                match group_mode:
                    case MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL:
                        group_categorical_mask |= new_index_mask
                        group_values_mask |= new_index_mask
                    case MeasIndexGroupOptions.CATEGORICAL_ONLY:
                        group_categorical_mask |= new_index_mask
                    case MeasIndexGroupOptions.NUMERICAL_ONLY:
                        group_values_mask |= new_index_mask
                    case _:
                        raise ValueError(f"Invalid group mode: {group_mode}")

            categorical_masks.append(group_categorical_mask)
            numerical_masks.append(group_values_mask)

        return torch.stack(categorical_masks, dim=-2), torch.stack(numerical_masks, dim=-2)

    def dynamic_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        batch_size, sequence_length, num_data_elements = batch["dynamic_values_mask"].shape
        out_shape = (batch_size, sequence_length, self.out_dim)

        if self.split_by_measurement_indices:
            categorical_mask, numerical_mask = self.split_batch_into_measurement_index_buckets(batch)
            _, _, num_measurement_buckets, _ = categorical_mask.shape
            out_shape = (batch_size, sequence_length, num_measurement_buckets, self.out_dim)

            expand_shape = (
                batch_size,
                sequence_length,
                num_measurement_buckets,
                num_data_elements,
            )
            indices = batch["dynamic_indices"].unsqueeze(-2).expand(*expand_shape)
            values = batch["dynamic_values"].unsqueeze(-2).expand(*expand_shape)
            measurement_indices = batch["dynamic_measurement_indices"].unsqueeze(-2).expand(*expand_shape)
            values_mask = batch["dynamic_values_mask"].unsqueeze(-2).expand(*expand_shape)
            values_mask = values_mask & numerical_mask
        else:
            indices = batch["dynamic_indices"]
            values = batch["dynamic_values"]
            measurement_indices = batch["dynamic_measurement_indices"]
            values_mask = batch["dynamic_values_mask"]
            categorical_mask = None

        indices_2D = indices.reshape(-1, num_data_elements)
        values_2D = values.reshape(-1, num_data_elements)
        meas_indices_2D = measurement_indices.reshape(-1, num_data_elements)
        values_mask_2D = values_mask.reshape(-1, num_data_elements)
        if categorical_mask is not None:
            categorical_mask_2D = categorical_mask.reshape(-1, num_data_elements)
        else:
            categorical_mask_2D = None

        embedded = self.embed(indices_2D, meas_indices_2D, values_2D, values_mask_2D, categorical_mask_2D)

        # Reshape back out into the original shape. Testing ensures these reshapes reflect original structure
        return embedded.view(*out_shape)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        if batch.batch_size == 0:
            return torch.empty(0, batch.sequence_length, self.out_dim)
        elif batch.sequence_length == 0:
            return torch.empty(batch.batch_size, 0, self.out_dim)

        if batch.n_data_elements == 0:
            embedded = torch.zeros(
                batch.batch_size,
                batch.sequence_length,
                self.out_dim,
                device=batch["dynamic_indices"].device,
            )
        else:
            embedded = self.dynamic_embedding(batch)
        # embedded is of shape (batch_size, sequence_length, out_dim) or of shape
        # (batch_size, sequence_length, num_measurement_buckets, out_dim)

        if batch.event_mask is not None:
            mask = batch.event_mask
            while len(mask.shape) < len(embedded.shape):
                mask = mask.unsqueeze(-1)

            mask = mask.expand_as(embedded)
            embedded = torch.where(mask, embedded, torch.zeros_like(embedded))

        if self.static_embedding_mode == StaticEmbeddingMode.DROP:
            return embedded

        if batch.n_static_data_elements == 0:
            static_embedded = torch.zeros(
                batch.batch_size,
                self.out_dim,
                device=batch["static_indices"].device,
            )
        else:
            static_embedded = self.static_embedding(batch)
        # static_embedded is of shape (batch_size, out_dim)

        static_embedded = static_embedded.unsqueeze(1)
        # static_embedded is now of shape (batch_size, 1, out_dim)

        if self.split_by_measurement_indices:
            static_embedded = static_embedded.unsqueeze(2)
            # static_embedded is now of shape (batch_size, 1, 1, out_dim)

        match self.static_embedding_mode:
            case StaticEmbeddingMode.SUM_ALL:
                embedded = self.dynamic_weight * embedded + self.static_weight * static_embedded
                if batch.event_mask is not None:
                    return torch.where(mask, embedded, torch.zeros_like(embedded))
            case _:
                raise ValueError(f"Invalid static embedding mode: {self.static_embedding_mode}")
