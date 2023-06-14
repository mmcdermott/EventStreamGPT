## Using the Dataset

Now that the dataset is pre-built, we can use it. The dataset can be used directly (not through the PyTorch
Dataset format) for several applications. Here, we highlight two:

1. Dataset exploration & visualization.
2. Building task dataframes for fine-tuning. In this step, we'll also show how to craft zero-shot labelers,
   though this doesn't really rely on the dataset class.

### Dataset Exploration & Visualization

Event Stream GPT comes with some pre-built utilities to aid in exploring and understanding datasets, through
the `visualize` and `describe` methods. Calling these methods on a freshly re-loaded MIMIC-IV cohort yields
the following:

```{literalinclude} dataset_description.txt
```

```{eval-rst}
.. subfigure:: AB|CD
   :gap: 8px
   :subcaptions: below
   :name: sample_visualizations
   :class-grid: outline

   .. image:: dataset_visualizations/perc_subj_with_event_by_age.png
      :alt: % of subjects with an event by age and gender.

   .. image:: dataset_visualizations/events_per_subj.png
      :alt: Total number of events per subject, by gender.

   .. image:: dataset_visualizations/events_per_subj_by_age.png
      :alt: Events per subject at age, by gender.

   .. image:: dataset_visualizations/subject_gender.png
      :alt: Subject gender breakdown

   Sample visualizations over the MIMIC-IV cohort.
```

### Building Task DataFrames (& Labelers!)

In order to assess performance on downstream tasks over these data, we need to define "task dataframes" that
describe these downstream targets. This takes two forms: first, a simple task dataframe which defines a task
schema and cohort, and second, a zero-shot labeling function that can infer empirical task labels from
generated batches. We describe both options here.

#### Task DataFrames

Task dataframes in our setting are built using Polars queries over the internal dataframes of the dataset
object. These can be seen in the tutorial repository notebook,
[here](https://github.com/mmcdermott/MIMICIV_FMs_public/blob/main/notebooks/Build%20Task%20DataFrames.ipynb).

The logic to construct these task dataframes is relatively simple, and we hope to soon add functionality to
allow these to be configurable without needing to write explicit code. In this example, we define two tasks:
30-day readmission risk prediction and in-hospital mortality prediction. Both can be found in the notebook
linked above, but we will show a sample here demonstrating the construction, and final schema, of the task
dataframe for readmission risk prediction below.

```{literalinclude} readmission_risk_stats.py
---
language: python
---
```

#### Zero-shot Labelers

Zero-shot labelers are user-defined functors that can compute an empirical label for a generated batch of
data, to enable zero-shot evaluation of an Event Stream GPT. These are fully documented in the
[source code](https://github.com/mmcdermott/MIMICIV_FMs_public/tree/main/task_labelers) for this working
example, but we also highlight one example labeler below, for in-hospital mortality prediction.

```{literalinclude} in_hosp_mort_labeler.py
---
language: python
---
```

After defining these labelers, one simply needs to copy them into the task dataframes folder for the
corresponding dataset, and they can be used for evaluation with no issue.
