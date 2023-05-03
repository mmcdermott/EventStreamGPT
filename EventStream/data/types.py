import dataclasses
import enum
from typing import Dict, Optional, Union

import torch

from ..utils import StrEnum


class InputDFType(StrEnum):
    STATIC = enum.auto()
    EVENT = enum.auto()
    RANGE = enum.auto()


class InputDataType(StrEnum):
    CATEGORICAL = enum.auto()
    FLOAT = enum.auto()
    TIMESTAMP = enum.auto()
    BOOLEAN = enum.auto()


@dataclasses.dataclass
class PytorchBatch:
    event_mask: Optional[torch.BoolTensor] = None
    time: Optional[torch.FloatTensor] = None

    static_indices: Optional[torch.LongTensor] = None
    static_measurement_indices: Optional[torch.LongTensor] = None

    dynamic_indices: Optional[torch.LongTensor] = None
    dynamic_measurement_indices: Optional[torch.LongTensor] = None
    dynamic_values: Optional[torch.FloatTensor] = None
    dynamic_values_mask: Optional[torch.BoolTensor] = None

    stream_labels: Optional[Dict[str, Union[torch.FloatTensor, torch.LongTensor]]] = None

    @property
    def device(self):
        return self.event_mask.device

    @property
    def batch_size(self) -> int:
        return self.event_mask.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.event_mask.shape[1]

    @property
    def n_data_elements(self) -> int:
        return self.dynamic_indices.shape[2]

    @property
    def n_static_data_elements(self) -> int:
        return self.static_indices.shape[1]

    def __getitem__(self, item: str) -> torch.Tensor:
        return dataclasses.asdict(self)[item]

    def __setitem__(self, item: str, val: torch.Tensor):
        if not hasattr(self, item):
            raise KeyError(f"Key {item} not found")
        setattr(self, item, val)

    def items(self):
        return dataclasses.asdict(self).items()

    def keys(self):
        return dataclasses.asdict(self).items()

    def values(self):
        return dataclasses.asdict(self).items()


class TemporalityType(StrEnum):
    """Describes the different ways in which a measure can vary w.r.t. time. As a `StrEnum`, can be
    used interchangeably with the lowercase versions of the member name strings (e.g., `STATIC` is
    equivalent to `'static'`).

    Members:
        `STATIC` (`'static'`):
            This measure is static per-subject. Currently only supported with classificaton data modalities.

        `DYNAMIC` (`'dynamic'`):
            This measure is dynamic with respect to time in a general manner. It will be recorded potentially
            many times per-event, and can take on either categorical or partially observed regression data
            modalities.

        `FUNCTIONAL_TIME_DEPENDENT` (`'functional_time_dependent'`):
            This measure varies with respect to time and the static measures of a subject in a manner that can
            be pre-specified in known functional form. The "observations" of this measure will be computed on
            the basis of that functional form and added to the observed events. Currently only supported with
            categorical or fully observed regression variables.
    """

    STATIC = enum.auto()
    DYNAMIC = enum.auto()
    FUNCTIONAL_TIME_DEPENDENT = enum.auto()


class DataModality(StrEnum):
    """The modality of a data element, which dictates pre-processing, embedding, and possible
    generation of said element. As a `StrEnum`, can be used interchangeably with the lowercase
    versions of the member name strings (e.g., `DROPPED` is equivalent to `'dropped'`).

    TODO(mmd): Maybe missing:
        * PARTIALLY_OBSERVED_SINGLE_LABEL_CLASSIFICATION:
            A data element that may or may not be observed, but if observed takes on exactly one of a fixed
            set of classes.
            Examples: Laboratory test results taking the value of "categorical"

    Members:
        `DROPPED` (`'dropped'`): This column was dropped due to occurring too infrequently for use.

        `SINGLE_LABEL_CLASSIFICATION` (`'single_label_classification'`):
            This data modality must take on a single label in all possible instances in which it can be
            observed.
            This will never have an associated data value measured.
            Element will be generated via single-label, multi-class classification, only applied in valid
            instances (e.g., in events of the appropriate type).

        `MULTI_LABEL_CLASSIFICATION` (`'multi_label_classification'`):
            This data modality can occur zero or more times with different labels in valid instances.
            This will never have an associated data value measured (see MULTIVARIATE_REGRESSION).
            Element will be generated via multi-label, binary classification.

        `MULTIVARIATE_REGRESSION` (`'multivariate_regression'`):
            A column which can occur zero or more times per event with different labels and
            associated numerical values, keyed by label. All multivariate regression measures are assumed to
            be partially observed at present.
            Element keys will be generated via multi-label, binary classification.
            Values will be generated via probabilistic regression.

        `UNIVARIATE_REGRESSION` (`'univariate_regression'`):
            This column is a continuous-valued numerical measure which is fully observed across all
            dimensions. Currently only supported on static or time-dependent columns, and then only for
            univariate data.
    """

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    DROPPED = enum.auto()
    SINGLE_LABEL_CLASSIFICATION = enum.auto()
    MULTI_LABEL_CLASSIFICATION = enum.auto()
    MULTIVARIATE_REGRESSION = enum.auto()
    UNIVARIATE_REGRESSION = enum.auto()


class NumericDataModalitySubtype(StrEnum):
    """Numeric value types. Is used to characterize both entire measures (e.g., 'age' takes on
    integer values) or sub-measures (e.g., within the measure of "vitals signs", observations for
    the key "heart rate" take on float values). As a `StrEnum`, can be used interchangeably with
    the lowercase versions of the member name strings (e.g., `DROPPED` is equivalent to
    `'dropped'`).

    Members:
        `DROPPED` (`'dropped'`): The values of this measure (or sub-measure) were dropped.
        `INTEGER` (`'integer'`): This measure (or sub-measure) takes on integer values.
        `FLOAT` (`'float'`): This measure (or sub-measure) takes on floating point values.
        `CATEGORICAL_INTEGER` (`'categorical_integer'`):
            This formerly integer measure/sub-measure has been converted to take on categorical values.
            Options can be found in the global vocabulary, with the syntax f"{key_col}__EQ_{orig_val}".
        `CATEGORICAL_FLOAT` (`'categorical_float'`):
            This formerly floating point measure/sub-measure has been converted to take on categorical values.
            Options can be found in the global vocabulary, with the syntax f"{key_col}__EQ_{orig_val}".
    """

    DROPPED = enum.auto()
    INTEGER = enum.auto()
    FLOAT = enum.auto()
    CATEGORICAL_INTEGER = enum.auto()
    CATEGORICAL_FLOAT = enum.auto()
