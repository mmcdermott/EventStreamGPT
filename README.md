# EventStream
## Installation
Installation can be done via conda with the `env.yml` file: `conda env create -n ${ENV_NAME} -f env.yml`.

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
You can see examples of this codebase at work via the tests.

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

Please see the `EventStreamData` README.md file for more information.

## `EventStreamTransformer`
Functionally, there are three areas of differences between a traditional sequence transformer and an
`EventStreamTransformer`: the input, how attention is processed in a per-event manner, and how generative
output layers work. Please see `EventStreamTransformer`'s README file for more information.

# Testing
`EventStream` code is tested in the global `tests` folder. These tests can be run via `python -m unittest` in
the global directory. These tests are not exhaustive, particularly in covering the operation of
`EventStreamTransformer`, but they are relatively comprehensive over core `EventStreamData` functionality.
