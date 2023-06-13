import enum
from typing import Union

import torch

from ..utils import StrEnum
from .types import PytorchBatch


class EmbeddingMode(StrEnum):
    """The different ways that the data can be embedded."""

    JOINT = enum.auto()
    """Embed all data jointly via a single embedding layer, weighting observed measurement embdddings by
    values when present."""

    SPLIT_CATEGORICAL_NUMERICAL = enum.auto()
    """Embed the categorical observations of measurements separately from their numerical values, and combine
    the two via a specifiable strategy."""


class MeasIndexGroupOptions(StrEnum):
    """The different ways that the `split_by_measurement_indices` argument can be interpreted.

    If measurements are split, then the final embedding can be seen as a combination of
    ``emb_cat(measurement_indices)`` and ``emb_num(measurement_indices, measurement_values)``, where ``emb_*``
    are embedding layers with sum aggregations that take in indices to be embedded and possible values to use
    in the output sum. This enumeration controls how those two elements are combined for a given measurement
    feature.
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
    """This class efficiently embeds an `PytorchBatch` into a fixed-size embedding.

    This embeds the `PytorchBatch`'s dynamic and static indices into a fixed-size embedding via a PyTorch
    `EmbeddingBag` layer, weighted by the batch's ``dynamic_values`` (respecting ``dynamic_values_mask``).
    This layer assumes a padding index of 0, as that is how the `PytorchDataset` object is structured.
    layer, taking into account `dynamic_indices` (including an implicit padding index of 0), It *does not*
    take into account the time component of the events; that should be embedded separately.

    It has two possible embedding modes; a joint embedding mode, in which categorical data and numerical
    values are embedded jointly through a unified feature map, which effectively equates to a constant value
    imputation strategy with value 1 for missing numerical values, and a split embedding mode, in which
    categorical data and numerical values that are present are embedded through separate feature maps, which
    equates to an imputation strategy of zero imputation (equivalent to mean imputation given normalization)
    and indicator variables indicating present variables. This further follows (roughly) the embedding
    strategy of :footcite:t:`gorishniy2021revisiting` (`link`_) for joint embedding of categorical and
    multi-variate numerical features. In particular, given categorical indices and associated continuous
    values, it produces a categorical embedding of the indices first, then (with a separate embedding layer)
    re-embeds those categorical indices that have associated values observed, this time weighted by the
    associated numerical values, then outputs a weighted sum of the two embeddings. In the case that numerical
    and categorical output embeddings are distinct, both are projected into the output dimensionality through
    additional linear layers prior to the final summation.

    The model uses the joint embedding mode if categorical and numerical embedding dimensions are not
    specified; otherwise, it uses the split embedding mode.

    .. _link: https://openreview.net/pdf?id=i_Q1yrOegLY

    .. footbibliography::

    Args:
        n_total_embeddings: The total vocabulary size that needs to be embedded.
        out_dim: The output dimension of the embedding layer.
        static_embedding_mode: The way that static embeddings are combined with the dynamic embeddings.
        categorical_embedding_dim: The dimension of the categorical embeddings. If `None`, no separate
            categorical embeddings are used.
        numerical_embedding_dim: The dimension of the numerical embeddings. If `None`, no separate numerical
            embeddings are used.
        split_by_measurement_indices: If not `None`, then the `dynamic_indices` are split into multiple
            groups, and each group is embedded separately. The `split_by_measurement_indices` argument is a
            list of lists of indices. Each inner list is a group of indices that will be embedded separately.
            Each index can be an integer, in which case it is the index of the measurement to be embedded, or
            it can be a tuple of the form ``(index, meas_index_group_mode)``, in which case ``index`` is the
            index of the measurement to be embedded, and ``meas_index_group_mode`` indicates whether the group
            includes only the categorical index of the measurement, only the numerical value of the
            measurement, or both its categorical index and it's numerical values, as specified through the
            `MeasIndexGroupOptions` enum. Note that measurement index groups are assumed to only apply to the
            dynamic indices, not the static indices, as static indices are never generated and should be
            assumed to be causally linked to all elements of a given event. Furthermore, note that if
            specified, no measurement group **except for the first** can be empty. The first is allowed to be
            empty to account for settings where a model is built with a dependency graph with no
            `FUNCTIONAL_TIME_DEPENDENT` measures, as time is always assumed to be the first element of the
            dependency graph.
        do_normalize_by_measurement_index: If `True`, then the embeddings of each measurement are normalized
            by the number of measurements of that `measurement_index` in the batch.
        static_weight: The weight of the static embeddings. Only used if `static_embedding_mode` is not
            `StaticEmbeddingMode.DROP`.
        dynamic_weight: The weight of the dynamic embeddings. Only used if `static_embedding_mode` is not
            `StaticEmbeddingMode.DROP`.
        categorical_weight: The weight of the categorical embeddings. Only used if `categorical_embedding_dim`
            and `numerical_embedding_dim` are not `None`.
        numerical_weight: The weight of the numerical embeddings. Only used if `categorical_embedding_dim` and
            `numerical_embedding_dim` are not `None`.

    Raises:
        TypeError: If any of the arguments are of the wrong type.
        ValueError: If any of the arguments are not valid.

    Examples:
        >>> valid_layer = DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        >>> valid_layer.embedding_mode
        <EmbeddingMode.JOINT: 'joint'>
        >>> valid_layer = DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ...     categorical_embedding_dim=5,
        ...     numerical_embedding_dim=5,
        ...     split_by_measurement_indices=None,
        ...     do_normalize_by_measurement_index=False,
        ...     categorical_weight=1 / 2,
        ...     numerical_weight=1 / 2,
        ... )
        >>> valid_layer.embedding_mode
        <EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL: 'split_categorical_numerical'>
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim="10",
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `out_dim` must be an `int`.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=-10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: `out_dim` must be positive.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings="100",
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `n_total_embeddings` must be an `int`.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=-100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: `n_total_embeddings` must be positive.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ...     categorical_embedding_dim=5,
        ...     numerical_embedding_dim=5,
        ...     split_by_measurement_indices=[4, (5, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `split_by_measurement_indices` must be a list of lists.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ...     categorical_embedding_dim=5,
        ...     numerical_embedding_dim=5,
        ...     split_by_measurement_indices=[[4, [5, MeasIndexGroupOptions.CATEGORICAL_ONLY]]],
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `split_by_measurement_indices` must be a list of lists of ints and/or tuples.
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
        """Initializes the layer."""
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
            if (categorical_embedding_dim is None) or (numerical_embedding_dim is None):
                raise ValueError(
                    "Both `categorical_embedding_dim` and `numerical_embedding_dim` must not be `None` when "
                    f"self.embedding_mode = {self.embedding_mode}"
                )
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
        """Returns a normalization tensor for the measurements observed in the input, by row.

        Args:
            measurement_indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the
                indices of the measurements in each batch element. Zero indicates padded measurements and the
                returned mask will have a value of zero in those positions.

        Returns:
            A tensor of the same shape as the input where the value at position ``i, j`` is one divided by the
            number of times the measurement index at the position ``i, j`` in the input occurs in the input
            row ``i``, normalized such that each row sums to one. Said alternatively, this returns a tensor
            that assigns each unique measurement in the input total equal weight out of 1, then splits that
            total weight evenly among all occurrences of that measurement in the input.

        Examples:
            >>> import torch
            >>> measurement_indices = torch.LongTensor([[1, 2, 5, 2, 2], [1, 3, 5, 3, 0]])
            >>> DataEmbeddingLayer.get_measurement_index_normalziation(measurement_indices)
            tensor([[0.3333, 0.1111, 0.3333, 0.1111, 0.1111],
                    [0.3333, 0.1667, 0.3333, 0.1667, 0.0000]])
        """

        one_hot = torch.nn.functional.one_hot(measurement_indices)

        normalization_vals = 1.0 / one_hot.sum(dim=-2)

        normalization_vals = torch.gather(normalization_vals, dim=-1, index=measurement_indices)

        # Make sure that the zero index is not counted in the normalization
        normalization_vals = torch.where(measurement_indices == 0, 0, normalization_vals)

        normalization_vals_sum = normalization_vals.sum(dim=-1, keepdim=True)
        normalization_vals_sum = torch.where(normalization_vals_sum == 0, 1, normalization_vals_sum)
        return normalization_vals / normalization_vals_sum

    def _joint_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns an embedding of the input indices, weighted by the values, if present.

        Args:
            indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the indices of the
                observations in the batch. Zero indicates padding.
            measurement_indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the
                indices of the measurements in each batch element. Zero indicates padding.
            values: A tensor of shape ``(batch_size, num_measurements)`` that contains any continuous values
                associated with the observations in the batch. If values are not present for an observation,
                the value in this tensor will be zero; however, a zero value itself does not indicate a value
                wasn't present for the observation.
            values_mask: A tensor of shape ``(batch_size, num_measurements)`` that contains a mask indicating
                whether a value was present for the observation in the batch.

        Returns:
            A tensor of shape ``(batch_size, out_dim)`` that contains the embedding of the input indices,
            weighted by their associated observed values, if said values are present. If values were not
            present with an observation, the embedding is unweighted for that observation (which corresponds
            to an implicit assumed value of **1** (not zero, which would eliminate that observation from the
            output). If `self.do_normalize_by_measurement_index`, the embeddings will also be weighted such
            that each unique measurement in the batch contributes equally to the output.
        """
        if values is None:
            values = torch.ones_like(indices, dtype=torch.float32)
        else:
            values = torch.where(values_mask, values, 1)

        if self.do_normalize_by_measurement_index:
            values *= self.get_measurement_index_normalziation(measurement_indices)

        return self.embed_layer(indices, per_sample_weights=values)

    def _split_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
        cat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a weighted sum of categorical and numerical embeddings of the input indices and values.

        This follows (roughly) the embedding strategy of :footcite:t:`gorishniy2021revisiting` (`link`_) for
        joint embedding of categorical and multi-variate numerical features. In particular, given categorical
        indices and associated continuous values, it produces a categorical embedding of the indices first,
        then (with a separate embedding layer) re-embeds those categorical indices that have associated values
        observed, this time weighted by the associated numerical values, then outputs a weighted sum of the
        two embeddings. In the case that numerical and categorical output embeddings are distinct, both are
        projected into the output dimensionality through additional linear layers prior to the final
        summation.

        .. _link: https://openreview.net/pdf?id=i_Q1yrOegLY

        .. footbibliography::

        Args:
            indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the indices of the
                observations in the batch. Zero indicates padding.
            measurement_indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the
                indices of the measurements in each batch element. Zero indicates padding.
            values: A tensor of shape ``(batch_size, num_measurements)`` that contains any continuous values
                associated with the observations in the batch. If values are not present for an observation,
                the value in this tensor will be zero; however, a zero value itself does not indicate a value
                wasn't present for the observation.
            values_mask: A boolean tensor of shape ``(batch_size, num_measurements)`` that contains a mask
                indicating whether or not a given index, value pair should be included in the numerical
                embedding.
            cat_mask: A boolean tensor of shape ``(batch_size, num_measurements)`` that contains a mask
                indicating whether or not a given index should be included in the categorical embedding.

        Returns:
            A tensor of shape ``(batch_size, out_dim)`` that contains the final, combined embedding of the
            input, in line with the description above.
        """
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

    def _embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
        cat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Either runs `_joint_embed` or `_split_embed` depending on `self.embedding_mode`.

        Args:
            indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the indices of the
                observations in the batch. Zero indicates padding.
            measurement_indices: A tensor of shape ``(batch_size, num_measurements)`` that contains the
                indices of the measurements in each batch element. Zero indicates padding.
            values: A tensor of shape ``(batch_size, num_measurements)`` that contains any continuous values
                associated with the observations in the batch. If values are not present for an observation,
                the value in this tensor will be zero; however, a zero value itself does not indicate a value
                wasn't present for the observation.
            values_mask: A boolean tensor of shape ``(batch_size, num_measurements)`` that contains a mask
                indicating whether or not a given index, value pair should be included in the numerical
                embedding.
            cat_mask: A boolean tensor of shape ``(batch_size, num_measurements)`` that contains a mask
                indicating whether or not a given index should be included in the categorical embedding.

        Returns:
            A tensor of shape ``(batch_size, out_dim)`` that contains the final embeddings of the
            input, in line with either `_joint_embed` or `_split_embed`.

        Raises:
            AssertionError: If `indices.max()` is greater than or equal to `self.n_total_embeddings`.
            ValueError: If `self.embedding_mode` is not a valid `EmbeddingMode`.
        """
        torch._assert(
            indices.max() < self.n_total_embeddings,
            f"Invalid embedding! {indices.max()} >= {self.n_total_embeddings}",
        )
        match self.embedding_mode:
            case EmbeddingMode.JOINT:
                return self._joint_embed(indices, measurement_indices, values, values_mask)
            case EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL:
                return self._split_embed(indices, measurement_indices, values, values_mask, cat_mask)
            case _:
                raise ValueError(f"Invalid embedding mode: {self.embedding_mode}")

    def _static_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        """Returns the embedding of the static features of the input batch.

        Args:
            batch: The input batch to be embedded.
        """
        return self._embed(batch["static_indices"], batch["static_measurement_indices"])

    def _split_batch_into_measurement_index_buckets(
        self, batch: PytorchBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the batch into groups of measurement indices.

        Given a batch of data and the list of measurement index groups passed at construction, this function
        produces categorical and numerical values masks for each of the measurement groups. The measurement
        groups (except for the first, which is reserved for `FUNCTIONAL_TIME_DEPENDENT` measurements, which
        may not be present) must not be empty.

        Args:
            batch: A batch of data.

        Returns:
            A tuple of tensors that contain the categorical mask and values mask for each group.

        Raises:
            ValueError: if either there is an empty measurement group beyond the first or there is an invalid
                specified group mode.
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

    def _dynamic_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        """Returns the embedding of the dynamic features of the input batch.

        Args:
            batch: The input batch to be embedded.
        """
        batch_size, sequence_length, num_data_elements = batch["dynamic_values_mask"].shape
        out_shape = (batch_size, sequence_length, self.out_dim)

        if self.split_by_measurement_indices:
            categorical_mask, numerical_mask = self._split_batch_into_measurement_index_buckets(batch)
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

        embedded = self._embed(indices_2D, meas_indices_2D, values_2D, values_mask_2D, categorical_mask_2D)

        # Reshape back out into the original shape. Testing ensures these reshapes reflect original structure
        return embedded.view(*out_shape)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        """Returns the final embeddings of the values in the batch.

        Args:
            batch: The input batch to be embedded.

        Returns:
            The final embeddings. These will either be of shape (batch_size, sequence_length, out_dim) or
            (batch_size, sequence_length, num_measurement_buckets, out_dim) depending on whether the
            measurements are split or not.

        Raises:
            AssertionError: If `indices.max()` is greater than or equal to `self.n_total_embeddings`.
            ValueError: If `self.embedding_mode` is not a valid `EmbeddingMode`, or if
                `split_by_measurement_indices` is not `None` and there either there is an empty measurement
                group beyond the first or there is an invalid specified group mode.

        Examples:
            >>> import torch
            >>> # Here we construct a batch with batch size of 2, sequence length of 3, number of static data
            >>> # elements of 3, and number of dynamic data elements of 2.
            >>> batch = PytorchBatch(
            ...     event_mask=torch.BoolTensor([[True, True, True], [True, True, False]]),
            ...     static_indices=torch.LongTensor([[1, 2, 3], [4, 5, 6]]),
            ...     static_measurement_indices=torch.LongTensor([[1, 1, 2], [2, 2, 3]]),
            ...     dynamic_indices=torch.LongTensor([[[7, 8], [11, 10], [8, 7]], [[8, 7], [8, 10], [0, 0]]]),
            ...     dynamic_measurement_indices=torch.LongTensor(
            ...         [[[4, 4], [5, 5], [4, 4]], [[4, 4], [4, 5], [0, 0]]]
            ...     ),
            ...     dynamic_values=torch.FloatTensor(
            ...         [[[1, 2], [0, 0], [1.1, 2.1]], [[5, 6], [7, 0], [0, 0]]]
            ...     ),
            ...     dynamic_values_mask=torch.BoolTensor(
            ...         [
            ...             [[True, True], [False, False], [True, True]],
            ...             [[True, True], [True, False], [False, False]],
            ...         ]
            ...     ),
            ... )
            >>> L = DataEmbeddingLayer(
            ...     n_total_embeddings=100,
            ...     out_dim=10,
            ...     static_embedding_mode=StaticEmbeddingMode.DROP,
            ...     categorical_embedding_dim=5,
            ...     numerical_embedding_dim=5,
            ...     split_by_measurement_indices=None,
            ...     do_normalize_by_measurement_index=False,
            ...     categorical_weight=1 / 2,
            ...     numerical_weight=1 / 2,
            ... )
            >>> out = L(batch)
            >>> out.shape # batch, seq_len, out_dim
            torch.Size([2, 3, 10])
            >>> L = DataEmbeddingLayer(
            ...     n_total_embeddings=100,
            ...     out_dim=10,
            ...     static_embedding_mode='sum_all',
            ...     categorical_embedding_dim=5,
            ...     numerical_embedding_dim=5,
            ...     split_by_measurement_indices=[
            ...         [(4, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
            ...         [5, (4, 'categorical_and_numerical')],
            ...     ],
            ...     do_normalize_by_measurement_index=True,
            ...     static_weight=1/3,
            ...     dynamic_weight=2/3,
            ...     categorical_weight=1/4,
            ...     numerical_weight=3/4,
            ... )
            >>> out = L(batch)
            >>> out.shape # batch, seq_len, dependency graph length (split_by_measruement_indices), out_dim
            torch.Size([2, 3, 2, 10])
        """
        embedded = self._dynamic_embedding(batch)
        # embedded is of shape (batch_size, sequence_length, out_dim) or of shape
        # (batch_size, sequence_length, num_measurement_buckets, out_dim)

        mask = batch.event_mask
        while len(mask.shape) < len(embedded.shape):
            mask = mask.unsqueeze(-1)

        mask = mask.expand_as(embedded)
        embedded = torch.where(mask, embedded, torch.zeros_like(embedded))

        if self.static_embedding_mode == StaticEmbeddingMode.DROP:
            return embedded

        static_embedded = self._static_embedding(batch).unsqueeze(1)
        # static_embedded is of shape (batch_size, 1, out_dim)

        if self.split_by_measurement_indices:
            static_embedded = static_embedded.unsqueeze(2)
            # static_embedded is now of shape (batch_size, 1, 1, out_dim)

        match self.static_embedding_mode:
            case StaticEmbeddingMode.SUM_ALL:
                embedded = self.dynamic_weight * embedded + self.static_weight * static_embedded
                return torch.where(mask, embedded, torch.zeros_like(embedded))
            case _:
                raise ValueError(f"Invalid static embedding mode: {self.static_embedding_mode}")
