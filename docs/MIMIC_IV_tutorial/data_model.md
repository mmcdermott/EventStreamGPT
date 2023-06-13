## High-level Data Model

Before we detail the usage of this pipeline, we need to cover the general data model of the pipeline on the
whole, which is illustrated in Figure 1. This data model can be broken down into three sections:

1. The core assumptions and internal data layout of the EFGPT pipeline
2. The pre-processing conventions and steps.
3. The final, deep-learning focused representation format and associated PyTorch Dataset data model.

We'll walk through each of those in detail here.

```{figure} Data_Overview.svg
---
width: 85%
alt: The general data model of the EFGPT data pipeline
align: center
---
**Figure 1**: The general data model of the EFGPT data pipeline.
```

### Assumptions & Internal Data Layout

The EFGPT pipeline data model is composed of three entities, each of which are tracked internally in a
separate dataframe.

#### Subjects

_Subjects_ (_e.g._, patients) are data owners. Information at the per-subject level is non-time-varying. In
Figure 1 (a), the sample patient record shown corresponds to a single subject $S_1$; that subject has just one
row in the `subjects_df` in Figure 1 (b).

#### Events

An _event_ is an instance of something happening to a subject at a specific timestamp. Events are unique at
the subject-timestamp level (_i.e._, no two events happen at the exact same time for the same subject). With
the exception of a sentinel, categorical, "event type" variable, events do not have specially encoded
information at the per-event level. Instead, events link in a one-to-many format to _dynamic measurements_.
The first three visits of the subject in Figure 1 (a) each correspond to a single event (as all components of
those visits are reported at the same timestamp in this example). As such, they each occupy a single row in
the `events_df` dataframe in Figure 1 (b).

#### Dynamic Measurements

In EFGPT, _Measurements_, in general, refers to any observation or recorded metric about a subject. They can
be static, in which case they are recorded at the per-subject level, not time-varying, and stored in the
subjects dataframe, or they can be _dynamic_ in which case they can occur arbitrarily in time and are recorded
in a separate dataframe. Any observation that is recorded at a subject's event is recorded as a row in the
dynamic measurements dataframe, linked to events through an event ID. This allows us to maintain a sparse data
structure and a minimal memory footprint overall, sa then other per-event details (e.g., event type, subject
ID, and timestamp) do not need to be repeated if a single event has many associated dynamic measurements. In
Fiure 1 (a), The various diagnostic codes, laboratory tests, procedures, etc. recorded in each of the first
three visits will all be recorded as separate measurements, and occupy unique rows in
`dynamic_measurements_df` in Figure 1 (b).

### Pre-processing

During pre-processing, the EFGPT pipeline, in general, performances the following steps:

1. Converts input data types into minimal memory equivalents (e.g., strings to categorical data types,
   64-bit signed integers for ID-spaces into \*-bit unsigned integer types, etc.).
2. Applies pre-set censoring, outlier removal, and filtering over infrequently observed measurements to
   limit the input space.
3. Fits measurement vocabularies, outlier detection parameters, and normalization parameters over the
   categorical and numerical values observed in the train set.
4. Universally filters out infrequently observed categorical variables and outliers, normalizes numerical
   variables, and converts categorical variables to indices.
5. Produces deep-learning friendly representations for downstream use via the [PytorchDataset](<>) class.

In this way, we can view the input of the entire EFGPT pipeline as the raw, pre-extraction input dataset, and
the output as a pre-cached PyTorch Dataset ready-made for deep-learning use.

### Deep-learning Representations

The deep learning representation is a polars dataframe written to disk. This dataframe has one row per
subject, a set of sentinel columns that contain all observed information about each subject in a highly sparse
format. In particular, this dataframe contains the following columns:

- `subject_id`: This column will be an unsigned integer type, and will have the ID of the subject
  for each row.
- `start_time`: This column will be a `datetime` type, and will contain the start time of the
  subject's record.
- `static_indices`: This column is a ragged, sparse representation of the categorical static
  measurements observed for this subject. Each element of this column will itself be a list of
  unsigned integers corresponding to indices into the unified vocabulary for the static measurements
  observed for that subject.
- `static_measurement_indices`: This column corresponds in shape to `static_indices`, but contains
  unsigned integer indices into the unified measurement vocabulary, defining to which measurement each
  observation corresponds. It is of the same shape and of a consistent order as `static_indices.`
- `time`: This column is a ragged array of the time in minutes from the start time at which each
  event takes place. For a given row, the length of the array within this column corresponds to the
  number of events that subject has.
- `dynamic_indices`: This column is a doubly ragged array containing the indices of the observed
  values within the unified vocabulary per event per subject. Each subject's data for this column
  consists of an array of arrays, each containing only the indices observed at each event.
- `dynamic_measurement_indices` This column is a doubly ragged array containing the indices of the
  observed measurements per event per subject. Each subject's data for this column consists of an
  array of arrays, each containing only the indices of measurements observed at each event. It is of
  the same shape and of a consistent order as `dynamic_indices`.
- `dynamic_values` This column is a doubly ragged array containing the indices of the
  observed measurements per event per subject. Each subject's data for this column consists of an
  array of arrays, each containing only the indices of measurements observed at each event. It is of
  the same shape and of a consistent order as `dynamic_indices`.
