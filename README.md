# EventStream
## Installation
  1. Install a conda environment via the `env.yml` file. It will complain about pip failures; this is
     expected: `conda env create -n ${ENV_NAME} -f env.yml`
  2. Activat ethe new environment: `conda activate ${ENV_NAME}`
  3. Install this github via the instructions in the repository's README:
     `https://github.com/mmcdermott/ifl-tpp`
  4. Install other pip dependencies: `pip install wandb transformers torchmetrics ml-mixins millify dill==0.3.6`

## Overview
This codebase contains utilities for working with _event stream_ datasets---meaning datasets where any given
sample $\boldsymbol x$ consists of a sequence of continuous-time (aka a _stream_) events, where each event can
consist of various categorical or continuous measurements of various structures.

For example, electronic health record (EHR) data is a prime example of _event stream_ data: any given
patient's medical record consists of a sequence of continuous-time medical interactions (visits, measurements,
phone calls, etc.), each of which may contain categorical diagnoses, continuous laboratory test results, or
even medical imaging or notes data (though the current incarnation of this dataset does not handle anything
other than categorical or continuous numerical per-event modalities).

To model these kinds of data, this codebase contains two major sub-modules: `EventStreamData`, which contains
classes and utiliteis for managing event stream datasets, both in raw form and in pytorch for modelling, and
`EventStreamTransformer`, which contains huggingface compatible transformer models for processing event stream
data, generative layers for performing marked point-process / generative, continuous-time sequence modelling
over event stream data, and lightning wrappers for training said models. To read more about these sub-modules
in detail, see the respective READMEs for their directories. However, some summary information is below.

## Examples
You can see examples of this codebase in the `./tests` folder to see how the `EventStreamDataset` and
`EventStreamTransformer` APIs work.

## `EventStreamData`
Event stream datasets are represented via a dataframe of events (containing event times, types, and subject
ids), subjects (containing subject ids and per-subject static measurements)  and per-event dynamic
measurements(containing event ids, types, subject ids, and arbitrary metadata columns). Many dynamic
measurements can belong to a single events. This class can also take in a functional specification for
measurements that can be computed in a fixed manner dependent only on event time and per-subject static data.

An `EventStreamDataset` can automatically pre-process train-set metadata, learning categorical vocabularies,
handling numerical data type conversion, rule-based outlier removal, and training of outlier detection and
normalization models.

It can also be processed into an `EventStreamPytorchDataset`, which represents these data via batches.

The batch representation has the following structure. Let us define the following variables:
  * `B` is the batch size.
  * `L` is the per-batch maximum sequence length.
  * `M` is the per-batch maximum number of data elements per event.
  * `K` is the per-batch maximum number of static data elements.
```
EventStreamPytorchBatch(**{
  # Control variables
  # These aren't used directly in actual computation, but rather are used to define losses, positional
  # embeddings, dependency graph positions, etc.
  'time': [B X L], # (FloatTensor, normalized such that the first entry for each sequence is 0)
  'event_type': [B X L], # (LongTensor, 0 <=> no event was observed)

  'dynamic_values_mask': [B X K], # (BoolTensor, indicates whether a static data element was observed)

  # Static Embedding Variables
  # These variables are static --- they are constant throughout the entire sequence of events.
  'static_indices': [B X K], # (LongTensor, 0 <=> no static data element was observed)
  'static_measurement_indices': [B X K], # (FloatTensor, 0 <= no static data element was observed)

  # Dynamic Embedding Variables
  # These variables are dynamic per-event.
  'dynamic_indices': [B X L X M], # (LongTensor, 0 <=> no dynamic data element was observed)
  'dynamic_values': [B X L X M], # (FloatTensor, 0 <= no dynamic data element was observed)
  'dynamic_measurement_indices': [B X L X M], # (LongTensor, 0 <=> no data element was observed
})
```

This encoding is efficient in that it does not store any data corresponding to unobserved elements, and it can
be efficiently embedded via a PyTorch `EmbeddingBag` operation, which takes into account indices and data
values.

## `EventStreamTransformer`
Functionally, there are three areas of differences between a traditional sequence transformer and an
`EventStreamTransformer`: the input, how attention is processed in a per-event manner, and how generative
output layers work.

### Input Layer
The input layer to this model must not only embed the contents of each event, but also take into account the
time-between events. Whereas a traditional transformer for sequences need only leverage ordinal position
embeddings to capture sequence-position, these models must account for the fact that time between events can
be highly variable.

Currently, this is simply handled with by adapting sinusoidal position embeddings to take as input continuous
time-since-start. This is likely a sub-optimal solution and other solutions like rotary position embeddings or
similar should be investigated.

### Attention Layer
Unlike a sequence model, a single event in an event stream model can have many sub-aspects, which may relate
to one another in a particular, causal, fashion. This can be reflected in an `EventStreamTransformer`, by
specifying a sequential order of `data_types` to process in that causal manner. This allows the model to not
try and generate subsequent events in a manner that assumes all aspects of the event are conditionally
independent from one another given the historical representation, but rather in a manner that assumes that
data types are dependent on those listed prior to them in the passed graph, even when conditioned on the
history.

### Output Layer
The output layer produces categorical distribution estimates for categorical variables (via logit scores per
vocabulary element) and numerical pytorch distribution objects for continuous variables, including
time-to-next-event. Currently, time-to-next-event can either be modeled via an exponential or a mixture of
lognormal distributions, and other continuous variables can only be modeled via output gaussian distributions.

# Testing
`EventStream` code is tested in the global `tests` folder of the repository. These tests can be run
via `python -m unittest` in the global directory. These tests are not exhaustive, particularly in
covering aspects of the operation of `EventStreamTransformer`.
