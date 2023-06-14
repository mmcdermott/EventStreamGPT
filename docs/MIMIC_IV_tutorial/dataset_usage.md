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

#### Task DataFrames

#### Zero-shot Labelers
