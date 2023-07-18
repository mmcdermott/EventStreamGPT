# Event Stream GPT

Event Stream GPT (ESGPT) is a library for streamlining the development of generative, pre-trained transformers (i.e., foundation models) over "event stream" datasets---datasets consisting of discrete sequences of complex events in continuous times (_aka_ multivariate temporal point processes). ESGPT is particularly motivated by _Electronic Health Record_ (EHR) datasets, which often consist of sequences of medical visits or events distributed in time, with any given event containing diverse laboratory test orders and results, medication prescriptions, diagnoses, procedures, etc.

ESGPT solves three critical problems to help drive research into foundation models over event stream modalities:

1. ESGPT provides a highly flexible, easy-to-use, and performant pipeline to extract, pre-process, and manage event stream datasets of a variety of types. With a simple configuration file, users can extract raw datasets from source, normalize and filter data per configurable rules, and compile deep-learning friendly, highly-sparse datasets for efficient generative modeling.
2. ESGPT provides a huggingface compatible modeling API built around these datasets that is generlizable across datasets, even when underlying data schemas differ. While models trained on one dataset are still not translatable to new datasets, within the ESGPT infrastructure, modeling code _is_ translateable across all datasets, making it dramatically easier to benchmark pre-training architectures and strategies.
3. ESGPT introduces critical capabilities into the modeling landscape of generative foundation models for these modalities, including the ability to naturally represent complex, intra-event causal dependencies and to define and measure zero-shot performance via a generative analog of prompting over these modalities.

Through these advantages, ESGPT will be an invaluable tool for anyone pursuing foundation models over EHR or
other forms of event stream data modalities. See the links below for more details.

## Contents

```{toctree}
---
glob:
maxdepth: 1
---
Overview <overview>
Usage Guide <usage>
MIMIC-IV Tutorial <MIMIC_IV_tutorial/index>
Local Data Tutorial <_collections/local_tutorial_notebook.ipynb>
License <license>
Module Reference <api/modules>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
