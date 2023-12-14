## Evaluating Pre-trained Models

We provide pre-built lightning modules for assessing baseline performance on fine-tuing tasks and for running
few-shot fine-tuning evaluation and zero-shot generative evaluation through user-defined labelers.

### Baseline Models

Event Stream GPT can also aid in producing baseline performance results across both traditional, scikit-learn
style models or via supervised training of ESGPT compatible neural network model architectures. For this
tutorial, we build both kinds of baselines, hyperparameter tuning a random forest scikit-learn pipeline which
can select between different kinds of historical aggregation windows, dimensionality reduction strategies,
imputation methods, scalers, and model parameters as well as a non-pre-trained transformer model across the
full parameter space therein.

The scikit-learn hyperparameter search pipeline is managed by
`scripts/launch_sklearn_baseline_supervised_wandb_hp_sweep.py`, which leverages the Hydra config
`configs/sklearn_models/sklearn_baseline_hyperparameter_sweep_base.yaml`. Similar to the pre-training
hyperparameter tuning script, users can leverage a provided sample weights and biases template report
[here](https://api.wandb.ai/links/mmd/f3k2okw8) to track the results of this hyperparameter search. For a
concrete example, in the linked report, embedded as an image below, we show that for predicting in hospital
mortality on the MIMIC-IV pipeline, a random forest pipeline can achieve up to at least 0.7 AUROC in this
setting.

```{eval-rst}
.. subfigure:: A
   :gap: 8px
   :subcaptions: below
   :name: sample_wandb_outputs
   :class-grid: outline

   .. image:: wandb_reports/sklearn_baseline_sweep.png
      :alt: Weights and biases reports for training a sklearn pipeline from scratch..

   Sample hyperparameter tuning weights and biases report for supervised sklearn models over the MIMIC-IV
   cohort.
```

Additionally, users can hyperparameter tune a neural network trained from scratch on just a supervised task
(for comparison against pre-trained/fine-tuned models). Such a search can be run using the
`scripts/launch_from_scratch_supervised_wandb_hp_sweep.py` script and
`configs/from_scratch_supervised_hyperparameter_sweep_base.yaml` config and tracked via template wandb report
[here](https://api.wandb.ai/links/mmd/wq6k5om0). Doing so for the readmission risk prediction task on our
sample MIMIC-IV pipeline shows that models can achieve peak tuning-set performance of an AUROC of 0.59.

### Few-shot performance

To fine-tune a model, use the `scripts/finetune.py` script. Much like pre-training, this script
leverages hydra to run, but now using the `FinetuneConfig` structured config object. To perform evaluation,
for our working example we can run the following command:

```bash
PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" \
	python $EVENT_STREAM_PATH/scripts/finetune.py \
	load_from_model_dir="$MODEL_DIR" \
	task_df_name="$TASK_NAME" \
	data_config.train_subset_size="$FT_SUBSET_SIZE" \
	data_config.train_subset_seed="$FT_SUBSET_SEED"
```

In this example, we ran this command for the two tasks discussed previously; 30-day readmission risk
prediction and in-hospital mortality prediction, with `FT_SUBSET_SIZE` set to 10, 50, and 250. After running
this command, the evaluation script will do the following:

1. Make a subdirectory to house model results, with the following syntax:
   `$MODEL_DIR/finetuning/run_specifier...`, where `run_specifier` is either `$TASK_NAME` for zero-shot
   runs (which are produced with a different script; see below) or
   `subset_size_$FT_SUBSET_SIZE/subset_seed_$FT_SUBSET_SEED/$TASK_NAME` for few-shot runs.
2. Fine-tune a model in those sub-directories, initializing from the pre-trained model's saved weights. This
   model is logged to weights and biases by default (though you may need to customize the project name).
   Note that, by default, this model will use the same hyperparameters as the pre-trained model from which
   it was initialized; this is unavoidable for architectural parameters, but is likely sub-optimal for
   regularization parameters.
3. Upon completion, outputs its final metrics to `tuning_metrics.json` and `held_out_metrics.json`.

When fine-tuning a model, there are still some hyperparameters that can be tuned on the specific task in
question, including dropout rates, batch sizes and learning rates, and weight decay parameters. We also
support hyperparameter tuning over these parameters via the `scripts/launch_finetuning_wandb_hp_sweep.py`
command, which uses the Hydra configuration file defined in
`configs/finetuning_hyperparameter_sweep_base.yaml` to run a weights and biases sweep over only these
modifiable fine-tuning parameters. Much like in pre-training and for the baseline models, there is a template
report users can leverage to analyze the results of these sweeps, available
[here](https://api.wandb.ai/links/mmd/479cunf7).

When we run this procedure for the Readmission Risk prediction task over MIMIC-IV, we find that fine-tuned
models are able to obtain peak tuning-set performance of an AUROC of 0.61, which is slightly higher than the
trained-from-scratch nueral network performance, suggesting that even on this small dataset, we do observe
some benefits from generative pre-training.

### Zero-shot Performance

Building on the existing HuggingFace API, you can also generate future values given a generative model very
easily and, through this, perform zero-shot evaluation. You can use generation to perform zero-shot
predictions for a given fine-tuning task by following the following steps:

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

For example, in this working example we provide lablers for both in-hospital mortality and readmission risk.
After copying these to the data cohort task directory, we can then run the following command to perform
zero-shot evaluation:

```bash
PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" \
	python $EVENT_STREAM_PATH/scripts/zeroshot.py \
	load_from_model_dir="$MODEL_DIR" \
	task_df_name="$TASK_NAME" \
	config.task_specific_params.num_samples=3 data_config.do_include_start_time_min=True \
	data_config.seq_padding_side=left data_config.max_seq_len=128
```

This code will execute the following steps:

1. Make a subdirectory to house model results, with the following syntax:
   `$MODEL_DIR/finetuning/$TASK_NAME`.
2. Iterate over the fine-tuning dataset and generate `task_specific_params.num_samples` samples off of each
   input, then use the labeler to assess the empirical labels and probabilitiy of an input being unpredictable
   for these generated samples.
3. Upon completion, store final metrics to `zero_shot_tuning_metrics.json` and
   `zero_shot_held_out_metrics.json`.

Note that zero-shot evaluation takes a non-trivial amount of time, as generating future samples for event
stream data is significantly more computationally expensive than generating traditional samples is, due to the
intra-event dependencies. Much like for few-shot fine-tuning, given our small dataset size, here we again see
negligible performance when running this command for the MIMIC-IV cohort and tasks. We do see, however, that
only a small fraction of events are unpredictable under this approach on real world data ($<5\%$ for mortality
prediction and $<0.5\%$ for readmission risk prediction), indicating this approach may be sufficiently robust
to be used in other settings.
