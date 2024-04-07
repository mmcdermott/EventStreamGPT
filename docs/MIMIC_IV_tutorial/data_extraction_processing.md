## Data Extraction & Pre-processing

Now that we know the overarching data model for the pipeline, let us explore how we can actually build a
dataset within it. The first entry point for our software is the data extraction and pre-processing component.
To detail this pipeline, we will walk through the user's process of creating a dataset and detail the
technical inner workings of the pipeline at each stage.

### Configuring the pipeline

The primary entry point for dataset building is through the `scripts/build_dataset.py` script. This script
uses hydra to manage configuration and arguments, with the default configuration file for the script being
found in `configs/dataset_base.yml`

Users can extend and specialize this configuration by defining their own yml file. To explain what the various
configuration options are used for, we will rely on an example configuration file that is suitable for our
working example over MIMIC-IV, shown below:

```{literalinclude} dataset_config.yml
---
language: yaml
---
```

With this configuration file saved to path `.../configs/dataset.yml`, and with `ESGPT_PATH` defined to point
to the root of the ESGPT repo, then the dataset pipeline can be built with the command

```bash
PYTHONPATH="$ESGPT_PATH:$PYTHONPATH" python \
	$ESGPT_PATH/scripts/build_dataset.py \
	--config-path=$(pwd)/configs \
	--config-name=dataset \
	"hydra.searchpath=[$ESGPT_PATH/configs]" [configuration args...]
```

The only mandatory command line configuration argument with this setup is the `cohort_name` argument. As can
be seen in the default `dataset_base.yml` configuration file in the base library, it's value is set to the
sentinel OmegaConf "MISSING" value, `???`, so must be overwritten either in the config file or on the command
line.

Working through this example, we can see there are several four key sections to this configuration file /
command:

#### Hydra-specific parameters

The `defaults:` block at the top is a Hydra specific inclusion, and ensures the script knows to merge this
configuration file. Similarly, the `hydra.searchpath=[$ESGPT_PATH/configs]` command line argument also ensures
Hydra knows to look for the base config in the ESGPT repository's configs path.

#### Inputs

This section of the config defines the input data sources from which raw data should be extracted. It consists
of two parts: a set of global parameters that are used across all input sources and a collection of specific
individual input sources, with configuration information detailing how to extract them.

##### Global Parameters

The global parameters, in this example, consist of the following:

```yaml
subject_id_col: subject_id
connection_uri: postgres://${oc.env:USER}:@localhost:5432/mimiciv

min_los: 3
min_admissions: 1
```

Of these, the `subject_id_col` is the only mandatory parameter, as it details what (single) ID column is
used to identify subjects uniquely across all input sources (i.e., all input sources must have a column with
this name that holds the unique ID of the subject). The `connection_uri` parameter is not mandatory, but it is
a software recognized keyword parameter that is used to provide the default connection URI for all database
queries in the config (if per-query URIs are not provided).

The remaining two parameters are custom parameters that are only used in the MIMIC-IV example, to make it
easier to configure on the fly. This is not a weakness of the configuration language; in fact, it is a
strength. _Any_ dataset can have dataset-specific configuration parameters in addition to the default one if
they make it easier to specify the dataset you want (in the bounds of the configuration file). For example,
through Hydra's interpolation syntax, we are able to use these parameters to control the cohort selection
query in the `patients` input block:

```yaml
patients:
  query: |-
    ... # Omitted for brevity
    WHERE subject_id IN (
      SELECT long_icu.subject_id FROM (
        (
          SELECT subject_id FROM mimiciv_icu.icustays WHERE los > ${min_los}
        ) AS long_icu INNER JOIN (
          SELECT subject_id
          FROM mimiciv_hosp.admissions
          GROUP BY subject_id
          HAVING COUNT(*) > ${min_admissions}
        ) AS many_admissions
    ... # Omitted for brevity
```

This allows us to overwrite those parameters in a given run on the command line, with, for example,
`... min_los=1 min_admissions=2`.

Note that these parameters drive the actual exclusion/inclusion criteria of subjects in our dataset. In
particular, these queries drop all subjects who don't have sufficient admissions or admissions with
sufficiently long ICU stays from the dataset, thereby filtering the full MIMIC-IV dataset down from the 300k
patients present in the full data to only the 12k patients who remain in our final cohort. Relatedly, the
query for laboratory test values, shown below, conjoins laboratory test names and their units of measure,
which can result in some laboratory test values being remapped to `UNK`s if they do not occur sufficiently
frequently with select units.

```yaml
labs:
  query:
    - |-
      SELECT subject_id, charttime, (itemid || ' (' || valueuom || ')') AS lab_itemid, valuenum FROM
      mimiciv_hosp.labevents
    - |-
      SELECT subject_id, charttime, (itemid || ' (' || valueuom || ')') AS lab_itemid, valuenum FROM
      mimiciv_icu.chartevents
  ts_col: charttime
```

##### Per-input Blocks:

The input-database specific input blocks define the raw datasets that we should read to build the output
dataset. They take the form of the nested keys and values within the `inputs` key. Each key defines a named
input data-table from which raw data will be extracted, and the value is a specific configuration object to
partially define that extraction process. Below is the full documentation of each parameter allowed in these
input blocks:

#### Measurements

The measurements block defines not from what input sources we should read, but what output measurements we
should include in our dataset. It has a very simple structure, consisting of a nested dictionary. The
outermost layer of this structure has a key for each valid temporal mode a measurement can take: `static`,
`dynamic`, or `functional_time_dependent`.

Within the `static` and `dynamic` keys, there is yet another nested dictionary, where the outer keys
correspond to the permitted measurement modalities: `single_label_classification`,
`multi_label_classification`, `univariate_regression`, and `multivariate_regression`. Within each of these,
there is one final dictionary, whose keys are the names of input sources from which measurements should be
pulled and whose values are the list of measurement names that should be extracted from said input source.
Note that these names are the _output_ names of the measurements, which are not necessarily the same as the
raw column names in the input dataset.

The `functional_time_dependent` key has a similar, but slightly different structure. Rather than having a
nested dictionary of measurements by input sources by modalities, it has an inner dictionary which has the
desired output `functional_time_dependent` measurement names as keys and whose values store the configuration
options for those measurement's configuration files.

#### Core configuration parameters

The core configuration block, reporduced below, contains any speciality `DatasetConfig` parameters.

```yaml
save_dir: ${oc.env:PROJECT_DATA_DIR}/${cohort_name}
outlier_detector_config:
  stddev_cutoff: 4.0
min_valid_vocab_element_observations: 25
min_valid_column_observations: 50
min_true_float_frequency: 0.1
min_unique_numerical_observations: 25
min_events_per_subject: 20
agg_by_time_scale: 2h
```

These parameters showcase several aspects of the configuration language. Firstly, the `save_dir` specification
showcases [Hydra/OmegaConfg's Interpolation Capabilities](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation).

Secondly, we can see in addition several other aspects of the configuration being specialized; this
configuration specifies that variables more than 4 standard deviations away from the mean should be considered
outliers, that columns/measurements must be observed 50 times to be included at all, that there must be at
least 10% of values actually being floating point for a numerical measure to not be re-cast as integer-valued,
for a numerical column to need at least 25 unique values to not be re-cast as categorical, to only retain
subjects who have at least 20 subjects, and to aggregate all events together into 2-hour buckets.

Outside of the core measurements and input blocks, any remaining parameters that are elements of the
[`DatasetConfig`](../api/EventStream.data.config.html) object will be incorporated into the final config and
reflected in pre-processing, etc.
