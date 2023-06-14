You can use several scripts from this repository. These scripts are built using
[hydra](https://hydra.cc/docs/intro/), so generally you will use them by specifying a mixture of command line
overrides and local configuration options in `yaml` files.

### Dataset Building

The script endpoint to build a dataset is in `scripts/build_dataset.py`. To run this script, simply call it
and override its parameters via hydra:

```bash
PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python \
	$EVENT_STREAM_PATH/scripts/build_dataset.py \
	--config-path=$(pwd)/configs \
	--config-name=dataset \
	"hydra.searchpath=[$EVENT_STREAM_PATH/configs]" arg1 arg2
```

In your local config file (which in the above command will be located at `$(pwd)/configs/dataset.yml`), you
specify the input sources, measurements to extract, and configuration parameters in accordance with
`configs/dataset_base.yml`. See the MIMIC-IV tutorial for examples.

### Pre-training

The script endpoint to launch a pre-training run, with the built in transformer model class here, is in
`scripts/pretrain.py`. To run this script, simply call it and override its parameters
via hydra:

```bash
PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python $EVENT_STREAM_PATH/scripts/pretrain.py \
	--config-path='/path/to/local/configs' \
	--config-name='local_config_name' \
	optimization_config.batch_size=24 optimization_config.num_dataloader_workers=64 # hydra overrides...
```

In your local config file (or via the command line), you can override various parameters, e.g.

```yaml
defaults:
  - pretrain_config # IMPORTANT: This defaults to the pre-defined repository config!
  - _self_

experiment_dir: /path/to/base/model/dir...

data_config:
  save_dir: /path/to/data/cohort

config:
  measurements_per_dep_graph_level:
    - [age, time_of_day]
    - [event_type]
    - [next_param, [multivariate_regression_task, categorical_only], "... 'can_do_multiline'",
      '...']
    -   - can_also_use_yaml_syntax
        - [multivariate_regression_task, categorical_and_numerical]
```

The default hydra config for this object is a structured config stored in the configstore with name
`pretrain_config`, defined in the `PretrainConfig` dataclass object in
[`generative_modeling.py`](https://github.com/mmcdermott/EventStreamGPT/blob/main/EventStream/transformer/lightning_modules/generative_modeling.py)
file:

```python
@hydra_dataclass
class PretrainConfig:
    do_overwrite: bool = False

    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
            **{
                k: v
                for k, v in StructuredTransformerConfig().to_dict().items()
                if k not in SKIP_CFG_PARAMS
            },
        }
    )
    optimization_config: OptimizationConfig = OptimizationConfig()
    data_config: PytorchDatasetConfig = PytorchDatasetConfig()
    metrics_config: MetricsConfig = MetricsConfig()

    experiment_dir: str = omegaconf.MISSING
    save_dir: str = "${experiment_dir}/pretrain/${now:%Y-%m-%d_%H-%M-%S}"

    wandb_name: str | None = "generative_event_stream_transformer"
    wandb_project: str | None = None
    wandb_team: str | None = None
    log_every_n_steps: int = 50

    num_dataloader_workers: int = 1

    do_detect_anomaly: bool = False
    do_final_validation_on_metrics: bool = True

    def __post_init__(self):
        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)
```

#### Hyperparameter Tuning

To launch a weights and biases hyperparameter sweep, you can use the
`scripts/launch_wandb_hp_sweep.py` file.

```bash
PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python $EVENT_STREAM_PATH/scripts/launch_wandb_hp_sweep.py \
	--config-path=/path/to/local/configs \
	--config-name=local_config_name \
	hydra.searchpath=[$EVENT_STREAM_PATH/configs] # This line ensures hydra can find the pre-defined default
```

An example of the overriding local config is:

```yaml
defaults:
  - hyperparameter_sweep_base # IMPORTANT: This defaults to the pre-defined repository config!
  - _self_

parameters:
  experiment_dir:
    value: /path/to/experiment/dir
  num_dataloader_workers:
    value: # of dataloader workers
  data_config:
    save_dir:
      value: /path/to/data/cohort
  config:
    measurements_per_dep_graph_level:
      values:
        -   - [param list 1 entry 1]
            - [param list 1 entry 2]
            - '...'
```

The default config establishes default ranges for a number of standard parameters, and uses hydra mandatory
missing parameters for other arguments that must be set on the command line. After you create the weights and
biases sweep, you can simply launch weights and biases agents, _in the directory `$EVENT_STREAM_PATH/scripts`
and sourced in the appropriate environment_ and models will spin up as normal.

During hyperparameter tuning, many warnings about "unnecessary parameters" may be printed -- e.g., `WARNING: categorical_embedding_dim is set to 16 but do_split_embeddings=False. Setting categorical_embedding_dim to None.` These are normal and do not indicate anything is wrong; rather, they merely reflect the fact that the
hyperparameter sweep will search over parameters even when they are made irrelevant by other choices in the
config.

#### Pre-training your own model

Of course, this module isn't merely intended for you to be able to run this class of models, but rather should
also enable you to easily test your own, huggingface API compatible models. To make your own model, you can
follow the below steps:

1. Write your own model class. Structure it to follow the interface of the pre-built model architectures.
2. Copy the
   [generative_modeling.py](https://github.com/mmcdermott/EventStreamGPT/blob/main/EventStream/transformer/lightning_modules/generative_modeling.py)
   file into your own repository. Adjust imports as necessary to refer to the installed EventStream Package
   and your new model.
3. Adjust the internals of your new lightning class so the internal model used is your model class.
4. Adjust the defined `PretrainConfig` object to point to your model's config class in the `_target_`
   variable. Rename the config so it does not conflict in the Hydra Store with the EventStream default
   `PretrainConfig`.
5. Copy whichever scripts you want to use from the repository, adjust them to point to your new lightning's
   train function and config by default, and launch models with these scripts as you would have previously
   with the built-in model.

On our roadmap of features to add includes support for dynamically defined user models and configs from the
command line with built in scripts out of the gate, so that fewer (if any) components need to be directly
copied from this repository; stay tuned for further updates on that front!

### Fine-tuning

To fine-tune a model, use the `scripts/finetune.py` script. Much like pre-training, this script
leverages hydra to run, but now using the `FinetuneConfig`
configuration object:

```python
@hydra_dataclass
class FinetuneConfig:
    load_from_model_dir: str | Path = omegaconf.MISSING
    seed: int = 1

    pretrained_weights_fp: Path | None = None
    save_dir: str | None = None

    do_overwrite: bool = False

    optimization_config: OptimizationConfig = OptimizationConfig()

    task_df_name: str | None = omegaconf.MISSING

    data_config_overrides: dict[str, Any] | None = dataclasses.field(
        default_factory=lambda: {
            "subsequence_sampling_strategy": SubsequenceSamplingStrategy.TO_END,
            "seq_padding_side": SeqPaddingSide.RIGHT,
        }
    )

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": None,
        }
    )

    task_specific_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "pooling_method": "last",
            "num_samples": None,
        }
    )

    config_overrides: dict[str, Any] = dataclasses.field(default_factory=lambda: {})

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "${task_df_name}_finetuning",
            "project": None,
            "team": None,
            "log_model": True,
            "do_log_graph": True,
        }
    )

    wandb_experiment_config_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "save_dir": "${save_dir}",
        }
    )
```

The hydra integration is not quite as smooth at fine-tuning time as it is during pre-training; namely, this is
because it the user may want to simultaneously load all prescient details from the pre-trained model
configuration setting and overwrite some details, such as dropout rates. This configuration object handles
much of this logic for you, and in general you will only need to specify (1) the directory of the pre-trained
model to load and fine-tune, (2) the name of the task dataframe (stored in the `task_dfs` subdirectory of the
dataset configuration file's `save_dir` parameter) for models to run successfully. In this case, a command may
look like:

```bash
PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python $EVENT_STREAM_PATH/scripts/finetune.py \
	load_from_model_dir=/pretrained/model/dir \
	optimization_config.batch_size=64 \
	optimization_config.init_lr=1e-4 \
	optimization_config.end_lr=null \
	optimization_config.max_epochs=25 \
	task_df_name=lv_ef/60d
```

If you wish to pursue a few-shot fine-tuning experiment, you can use the parameters `train_subset_size` and
`train_subset_seed` to control that.

### Zero-shot Generation

Building on the existing HuggingFace API, you can also generate future values given a generative model very
easily. In particular, given a `FinetuneConfig` object
describing the data/model you wish to use for generation, you can simply do the following:

```python
# Initialize the config, overwriting the `max_seq_len` argument to a smaller value for the `data_config` to
# account for the elements you'll generate within the model's maximum sequence length.
cfg = FinetuneConfig(
    load_from_model_dir=MODEL_DIR,
    task_df_name=TASK_DF_NAME,
    data_config_overrides={
        "max_seq_len": 128,
        "subsequence_sampling_strategy": "to_end",
        "do_include_start_time_min": True,
        "seq_padding_side": "left",
    },
)
ESD = Dataset.load(cfg.data_config.save_dir)
train_pyd = PytorchDataset(cfg.data_config, split="train")
M = ESTForGenerativeSequenceModeling.from_pretrained(
    cfg.pretrained_weights_fp, config=cfg.config
)
sample_dataloader = DataLoader(
    train_pyd, batch_size=1, collate_fn=train_pyd.collate, shuffle=False
)
sample_batch = next(iter(sample_dataloader))

generated = M.generate(
    sample_batch,
    max_new_events=2,  # Note that this must be within the model's `max_seq_len` - the input data length
    do_sample=True,
    return_dict_in_generate=True,
    output_scores=True,
)

# generated.batch contains an extended PytorchBatch object with both the original data and
# the new, generated data
```

#### Automated Zero-shot Evaluation

You can use generation to perform zero-shot predictions for a given fine-tuning task by following the
following steps:

1. Make a new python file which contains a "labeler": a python object subclassing the
   [`Labeler`](EventStream/transformer/zero_shot_labeler.py) interface which implements a `__call__` method
   taking as input a batch of data, an input sequence length, and a configuration file and predict your task's
   label. from that batch, if possible, and otherwise indicate that it is not possible. For example, if your
   task is to predict in-hospital mortality, your class would need to look at the elements of the batch after
   the input sequence length and see if any death events happen before the first discharge event.
2. You need to copy this labeling class definition file (all necessary functions and such used by the class
   must live in that single file) into the data directories task dataframes subfolder with the name
   `${task_df_name}_labeler.py`.
3. You can then use the `scripts/zeroshot.py` script to run a zero-shot evaluation via a Hydra
   config on that task labeler and any pre-trained model.
