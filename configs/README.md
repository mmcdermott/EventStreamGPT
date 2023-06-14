The Event Stream GPT package uses a mixture of yaml config files and structured configs. Structured configs
can be found in their various source files, and are documented therein. Here, we will discuss the default yaml
configs.

### Dataset Building Config

The default dataset building config (`configs/dataset_base.yaml`) is shown below; however, this config alone
is far from sufficient to build a dataset. To use this script successfully, the user must augment this config
with additional `inputs` and `measurements` blocks, which define the input sources from which the data
pipeline can read and the measurements that will be sourced from those datasets. See the MIMIC-IV tutorial for
more information.

```yaml
defaults:
  - outlier_detector_config: stddev_cutoff
  - normalizer_config: standard_scaler
  - _self_

cohort_name: ???
save_dir: ${oc.env:PROJECT_DIR}/data/${cohort_name}
subject_id_col: ???
seed: 1
split: [0.8, 0.1]
do_overwrite: false
DL_chunk_size: 20000
min_valid_vocab_element_observations: 25
min_valid_column_observations: 50
min_true_float_frequency: 0.1
min_unique_numerical_observations: 25
min_events_per_subject: 20
agg_by_time_scale:

hydra:
  job:
    name: build_${cohort_name}
  run:
    dir: ${save_dir}/.logs
  sweep:
    dir: ${save_dir}/.logs
```

### Hyperparameter Tuning Config

The default hyperparameter tuning config (`configs/hyperparameter_sweep_base.yaml`) defines a number of
pre-specified hyperparameter sweep choices for use. Users may override these defaults or use them in their
sweeps; however, users must override the `measurements_per_dep_graph_level` option as that varies per dataset
and cannot be set to an intelligent default.
