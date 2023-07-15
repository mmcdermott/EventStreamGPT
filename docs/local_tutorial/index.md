# Local Data Tutorial

In this tutorial, rather than running real models and configurations over MIMIC-IV, we'll work with a set of
local, synthetic files distributed with this repository, with the goal being to fully explore the details of
this pipeline. This tutorial will consist of both content on this page, running certain scripts on one's local
machine, and some jupyter notebooks. We will walk through the entire pipeline with these local examples and
discuss limitations of the pipeline, details of classes, scripts, etc.

## Synthetic Data

For this tutorial, we'll use the three synthetic data files distributed in the [sample_data/raw](<>) folder in
the repository:

```bash
[mmd:~/Projects/EventStreamGPT/sample_data/raw] [base] running_local_example(+6/-5)+* ± ls -lah
total 53M
drwxrwxr-x 2 mmd mmd 4.0K Jul 14 12:42 .
drwxrwxr-x 5 mmd mmd 4.0K Jul 14 16:39 ..
-rw-rw-r-- 1 mmd mmd 3.6M Jul 14 16:31 admit_vitals.csv
-rw-rw-r-- 1 mmd mmd  50M Jul 14 16:32 labs.csv
-rw-rw-r-- 1 mmd mmd 4.2K Jul 14 16:31 subjects.csv
```

```{note}
To see how those files are generated, look at [sample_data/Generate Synthetic Data.ipynb]().
```

These files contain the following data:

### `subjects.csv`

This file contains per-subject data. It has a subject identifier, a date of birth, a categorical static
measurement (eye color), and a continuous static measurement (height):

```{literalinclude} ../../sample_data/raw/subjects.csv
---
lines: 1-3
language: csv
---
```

This file is arranged with one row per subject.

### `admit_vitals.csv`

This file contains dynamic data quantifying both fictional subject hospital admissions, and fictional vitals
signs measured for those subjects.

```{literalinclude} ../../sample_data/raw/admit_vitals.csv
---
lines: 1-3
language: csv
---
```

Each row of this file records a unique vitals sign measurement for a patient, affiliated with the associated
admission listed in the row. This means that admission level information is _heavily duplicated_ within this
file, which is a phenomena sometimes observed in real data, and something we'll need to account for in our
pipeline's setup.

### `labs.csv`

This file contains dynamic data quantifying fictional subject laboratory test measurements.

```{literalinclude} ../../sample_data/raw/labs.csv
---
lines: 1-3
language: csv
---
```

Each row of this file contains a record of a particular lab test measured for a subject.

## Processing Synthetic Data with ESGPT

Now that we see the form of this synthetic data, we can examine how to process it with Event Stream GPT. From
the base directory of the ESGPT repository, we can run the following command:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH ./scripts/build_dataset.py \
	--config-path="$(pwd)/sample_data/" \
	--config-name=dataset \
	"hydra.searchpath=[$(pwd)/configs]"
```

You should see as output the printed line `Empty new events dataframe of type OUTPATIENT_VISIT!`, but
otherwise nothing. Before we proceed further, let's break down what this process has done, and how it could do
things differently. Clearly, the critical input to this pipeline is the dataset configuration file. But,
before we walk through this file, let's take a look at what the pipeline has produced.

### Inspecting the Output

The entire output dataset is stored in the [sample_data/processed/sample](<>) directory. Let's inspect its
contents:

```bash
[mmd:~/Projects/EventStreamGPT/sample_data/processed/sample] [base] running_local_example(+6/-5)+* ± ls -lah
total 19M
drwxrwxr-x 5 mmd mmd 4.0K Jul 14 16:32 .
drwxrwxr-x 3 mmd mmd 4.0K Jul 14 16:32 ..
-rw-rw-r-- 1 mmd mmd 1.8K Jul 14 16:32 config.json
drwxrwxr-x 2 mmd mmd 4.0K Jul 14 16:32 DL_reps
-rw-rw-r-- 1 mmd mmd  12M Jul 14 16:32 dynamic_measurements_df.parquet
-rw-rw-r-- 1 mmd mmd 5.2K Jul 14 16:32 E.pkl
-rw-rw-r-- 1 mmd mmd 7.0M Jul 14 16:32 events_df.parquet
-rw-rw-r-- 1 mmd mmd 1.5K Jul 14 16:32 hydra_config.yaml
-rw-rw-r-- 1 mmd mmd 2.3K Jul 14 16:32 inferred_measurement_configs.json
drwxrwxr-x 2 mmd mmd 4.0K Jul 14 16:32 inferred_measurement_metadata
-rw-rw-r-- 1 mmd mmd 1.7K Jul 14 16:32 input_schema.json
drwxrwxr-x 3 mmd mmd 4.0K Jul 14 16:32 .logs
-rw-rw-r-- 1 mmd mmd 2.7K Jul 14 16:32 subjects_df.parquet
-rw-rw-r-- 1 mmd mmd  771 Jul 14 16:32 vocabulary_config.json
[mmd:~/Projects/EventStreamGPT/sample_data/processed/sample] [base] running_local_example(+6/-5)+* ± du -sh .
30M     .
```

We can see that this directory contains a set of files and sub-directories, and that in total it takes up only
30 MB of disk space. Note that this is in contrast to the original, raw data, which took 53 MB on disk.

Each of these files contains different information about this synthetic dataset. Let's inspect them and see
what they contain. We'll go in a rough order that corresponds with where these files fit into the broader
pipeline.

#### Input & Logging Files

##### `hydra_config.yaml`

This file contains the full, resolved hydra input config to the dataset script. Whereas [`dataset.yaml`](<>)
(the input config file used in the script run above) relies on some default values in the built-in
[`dataset_base.yaml`](<>) config, the `hydra_config.yaml` is fully self-sufficient and can be used to reproduce
the pipeline run in its entirety. In this case, it is very similar to the [`dataset.yaml`](<>) file which we'll
inspect in more detail later, so we won't include it here.

##### `input_schema.json`

This file contains a processed version of the input data frame schemas used to read the raw data from disk. It
is produced from the input [`dataset.yaml`](<>) config file (which we'll discuss in more detail later), and is
stored in JSON format. This can be helpful to validate exactly what sources were read in what way. It is not
used in any downstream pipeline components, so is not essential to understand.

```{literalinclude} ../../sample_data/processed/sample/input_schema.json
---
language: json
---
```

##### `.logs`

This sub-directory contains a hydra run log file for this dataset build run. As the pipeline currently doesn't
take advantage of the python logging module at all, the files it contains are empty.

```bash
[mmd:~/Projects/EventStreamGPT/sample_data/processed/sample] [base] running_local_example(+6/-5)+* ± ls .logs/build_sample.log
.logs/build_sample.log
[mmd:~/Projects/EventStreamGPT/sample_data/processed/sample] [base] running_local_example(+6/-5)+* ± cat .logs/build_sample.log
```

#### Output Files

##### Dataset Configuration & Learned Measurement Metadata

###### `config.json`

This is the dataset's input config file. It contains the input measurement specifications and control
parameters for the dataset pipeline. This is largely set from the input `dataset.yaml` config.

###### `inferred_measurement_configs.json` & `inferred_measurement_metadata`

These represent the inferred pre-processing parameters for the inferred measurements. This is stored in two
forms: First, most data about the inferred measurements is stored in a flat JSON file,
`inferred_measurement_configs.json`. This file contains an object whose keys are measurement names and whose
values are configuration objects describing the measurements.

The full file looks like this:

```{literalinclude} ../../sample_data/processed/sample/inferred_measurement_configs.json
---
language: json
---
```

To isolate a single measurement, we can examine the configuration for `'eye_color'`:

```json
{
  "eye_color": {
    "name": "eye_color",
    "temporality": "static",
    "modality": "single_label_classification",
    "observation_frequency": 1.0,
    "functor": null,
    "vocabulary": {
      "vocabulary": [
        "UNK",
        "BROWN",
        "BLUE",
        "HAZEL",
        "GREEN"
      ],
      "obs_frequencies": [
        0.0,
        0.5125,
        0.2125,
        0.175,
        0.1
      ]
    },
    "values_column": null,
    "_measurement_metadata": null
  }
}
```

We can see that this configuration object details several facts about the eye color measurement:

- That it is a static, single-label classification measurement (these were specified in the input config)
- That this is observed on 100% of subjects.
- That the relative frequencies of the categories "Brown", "Blue", "Hazel", "Green" are 51.25%, 21.25%,
  17.5%, and 10%, respectively.

To see a different measurement, one that is a multivariate regression measurement, we can inspect the lab
tests measurement configs:

```json
{
  "lab_name": {
    "name": "lab_name",
    "temporality": "dynamic",
    "modality": "multivariate_regression",
    "observation_frequency": 0.9953452513588434,
    "functor": null,
    "vocabulary": {
      "vocabulary": [
        "UNK",
        "SpO2",
        "creatinine",
        "potassium",
        "SOFA__EQ_1",
        "GCS__EQ_1",
        "SOFA__EQ_2",
        "SOFA__EQ_3",
        "GCS__EQ_4",
        "GCS__EQ_3",
        "GCS__EQ_2",
        "GCS__EQ_5",
        "GCS__EQ_6",
        "GCS__EQ_7",
        "SOFA__EQ_4",
        "GCS__EQ_8",
        "GCS__EQ_9",
        "GCS__EQ_10",
        "GCS__EQ_11",
        "GCS__EQ_15",
        "GCS__EQ_12",
        "GCS__EQ_13",
        "GCS__EQ_14",
        "SOFA__EQ_1000000",
        "GCS__EQ_1000000"
      ],
      "obs_frequencies": [
        0.0,
        0.8259984895186395,
        0.04326148962598335,
        0.042245556731226326,
        0.027447439849105214,
        0.013256007422060696,
        0.01155863274600911,
        0.004522818236187147,
        0.0045065249727806655,
        0.004329215929827789,
        0.003943928171627486,
        0.0029251199950928526,
        0.0027363098250295197,
        0.0023232276763122785,
        0.0022705141770560182,
        0.0018171780834521783,
        0.0015440263145788287,
        0.001292918372667188,
        0.0010964407845302172,
        0.0009066721872076797,
        0.000854917115210624,
        0.0006613148088512674,
        0.0004907147567128245,
        5.750563555228412e-06,
        4.792136296023677e-06
      ]
    },
    "values_column": "lab_value",
    "_measurement_metadata": "/home/mmd/Projects/EventStreamGPT/sample_data/processed/sample/inferred_measurement_metadata/lab_name.csv"
  }
}
```

Here, in addition to the same information we see for eye color, we also see listed the associated values
column for this multivariate regression, and also a path to a measurement metadata object that contains more
statistics for this measurement. In addition, we can also see that under this configuration, the system has
expanded the two laboratory tests `'SOFA'` and `'GCS'` into categorical options.

We can inspect the detailed measurement metadata linked in this config object by looking at the csv file in
question.

```{literalinclude} ../../sample_data/processed/sample/inferred_measurement_metadata/lab_name.csv
---
language: csv
---
```

Within this file, we see a dataframe containing information about the different laboratory tests that the
system has processed, including their value type, information about the learned outlier model, and information
learned about their normalization variables. For example, the system has inferred that the GCS and SOFA scores
are categorical, integer variables, the SpO2 lab is an integer variable, and the potassium and creatinine labs
are floating point labs. Further, it has learned outlier bounds for the various continuous laboratory tests
and has fit the mean and standard deviation of the laboratory test values for these as well.

##### Processed DataFrames

###### `subjects_df.parquet`, `events_df.parquet`, `dynamic_measurements_df.parquet`

These files are the output, processed, internally represented versions of the raw input data, organized
according to the event-stream data model (see the Usage Guide for more information on that data model).

###### `DL_reps`

This directory contains the deep-learning formatted representtaions of the data. It is suitable for rapidly
iterating through batches of subject time-series, but less well suited towards querying and data manipulation.

```bash
[mmd:~/Projects/EventStreamGPT/sample_data/processed/sample] [base] running_local_example+ ± ls DL_reps/
held_out_0.parquet train_0.parquet tuning_0.parquet
```

##### Overall Class File

###### `E.pkl`

This file contains the class object itself and a collection of its attributes. It, importantly, _does not_
contain the nested dataframes (`subjects_df`, `events_df`, `dynamic_measurements_df`), as these are stored in
the files mentioned above and loaded lazily by the object during use, and similarly does not store the learned
measurement metadata files also mentioned above, which are lazily loaded in the same way.

```{warning}
Currently, this lazy saving/loading uses absolute paths, which makes transferring datasets to new locations
more challenging! Paths can be modified in the class and config object locally to fix this problem, but it is
a challenge.
```
