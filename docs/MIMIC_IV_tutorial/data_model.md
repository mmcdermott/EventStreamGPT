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

### Deep-learning Representations
