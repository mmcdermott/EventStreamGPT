## Pre-training Models

### Hyperparameter Tuning

#### Weights and Biases Sweep

The configuration file used in our working example can be found below. It specifies a total of 8 possible
options for `measurements_per_dep_graph_level`, specific to the MIMIC dataset. Otherwise, it relies on default
parameter selections from the `config/hyperparameter_sweep_base.yaml` file.

```{literalinclude} hyperparameter_sweep.yml
---
language: yaml
---
```

#### Template Analysis Report

A template hyperparameter sweep analysis report can be found
[here](https://wandb.ai/mmd/MIMIC_FMs_Public/reports/Hyperparameter-Tuning-Sweep--Vmlldzo0NjM3MDg1?accessToken=c5g4i8ba2solm7k92j0id9ihm3w9or0uuh50wshhuop42bcioksm0f40teeqd8yu).
Users can clone this into their own weights and biases projects to further accelerate hyperparameter tuning
analysis. Samples of its outputs can be found below.

```{eval-rst}
.. subfigure:: A|B
   :gap: 8px
   :subcaptions: below
   :name: sample_wandb_outputs
   :class-grid: outline

   .. image:: wandb_reports/hyperparameter_sweep_losses.png
      :alt: Parameter importance and loss curves plots.

   .. image:: wandb_reports/hyperparameter_sweep_comparisons.png
      :alt: More detailed breakdown of the impact of various parameters.

   Sample hyperparameter tuning weights and biases report graphs over the MIMIC-IV cohort.
```
