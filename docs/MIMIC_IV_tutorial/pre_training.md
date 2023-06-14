## Pre-training Models

### Hyperparameter Tuning

#### Weights and Biases Sweep

```{literalinclude} hyperparameter_sweep.yml
---
language: yaml
---
```

#### Template Analysis Report

A template hyperparameter sweep analysis report can be found [here](<>). Users can clone this into their own
weights and biases projects to further accelerate hyperparameter tuning analysis. Samples of its outputs can
be found below.

```{eval-rst}
.. subfigure:: A|B
   :gap: 8px
   :subcaptions: below
   :name: sample_visualizations
   :class-grid: outline

   .. image:: wandb_reports/hyperparameter_sweep_losses.png
      :alt: Parameter importance and loss curves plots.

   .. image:: wandb_reports/hyperparameter_sweep_comparisons.png
      :alt: More detailed breakdown of the impact of various parameters.

   Sample hyperparameter tuning weights and biases report graphs over the MIMIC-IV cohort.
```

### Pre-training Models over Subsets
