# EventStream

EventStream is a codebase for managing and modeling event stream datasets, which consist of sequences of continuous-time events containing various categorical or continuous measurements. Examples of such data include electronic health records, financial transactions, and sensor data. The repo contains two major sub-modules: EventStreamData, for handling event stream datasets in raw form and with Pytorch for modeling, and EventStreamTransformer, which includes Hugging Face-compatible transformer models, generative layers for marked point-process and continuous-time sequence modeling, and Lightning wrappers for training these models.

## Installation

Installation can be done via conda with the `env.yml` file:

```
conda env create -n ${ENV_NAME} -f env.yml

```

The `env.yml` file contains all the necessary dependencies for the system.

## Overview

This codebase contains utilities for working with event stream datasets, meaning datasets where any given sample consists of a sequence of continuous-time events. Each event can consist of various categorical or continuous measurements of various structures.

### EventStreamData

Event stream datasets are represented via a dataframe of events (containing event times, types, and subject ids), subjects (containing subject ids and per-subject static measurements), and per-event dynamic measurements (containing event ids, types, subject ids, and arbitrary metadata columns). Many dynamic measurements can belong to a single event. This class can also take in a functional specification for measurements that can be computed in a fixed manner dependent only on event time and per-subject static data.

An EventStreamDataset can automatically pre-process train-set metadata, learning categorical vocabularies, handling numerical data type conversion, rule-based outlier removal, and training of outlier detection and normalization models.

It can also be processed into an EventStreamPytorchDataset, which represents these data via batches.

Please see the EventStreamData `README.md` file for more information.

### EventStreamTransformer

Functionally, there are three areas of differences between a traditional sequence transformer and an EventStreamTransformer: the input, how attention is processed in a per-event manner, and how generative output layers work. Please see EventStreamTransformer's `README` file for more information.

## Examples

You can see examples of this codebase at work via the tests.

## Testing

EventStream code is tested in the global tests folder. These tests can be run via `python -m unittest` in the global directory. These tests are not exhaustive, particularly in covering the operation of EventStreamTransformer, but they are relatively comprehensive over core EventStreamData functionality.

## Contributing

Contributions to the EventStream project are welcome! If you encounter any issues, have feature requests, or would like to submit improvements, please follow these steps:

1. Open a new issue to report the problem or suggest a new feature.
2. Fork the repository and create a new branch for your changes.
3. Make your changes, ensuring that they follow the existing code style and structure.
4. Submit a pull request for review and integration into the main repository.

Please ensure that your contributions are well-documented and include tests where necessary.

## License

This project is licensed under the [LICENSE](LICENSE) file provided in the repository.
