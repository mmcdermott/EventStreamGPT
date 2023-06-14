This sub-module contains the following:

1. Two sample architectures for Event Stream GPTs: a _conditionally independent point process transformer_
   and a _nested attention point process transformer_.
2. Utility functions for building, autoregressively generating, and evaluating Event Stream GPTs.
3. Lightning modules with hydra supported entry functions for ease of training and evaluation.

### Sample Model Architectures

Functionally, across both pre-built model types, there are many aspects that are shared with classical
transformer architectures, and three major areas of differences between a traditional sequence transformer and
an `EventStreamTransformer`: the input, how attention is processed in a per-event
manner, and how generative output layers work. We will detail each point in this section across both pre-built
architectures.

#### Shared Components and General Model Flow

Both pre-built architectures follow similar model patterns. They take as input `PytorchBatch` objects, which
are then converted into continuous embeddings via a combination of data embedding layers and temporal position
embeddings. Next, these input embeddings are contextualized through a multi-layer transformer architecture.
Finally, the output embeddings for these events and their internal covariates (if applicable) are used to
predict subsequent events in a generative manner. The entire ensemble is then optimized by minimizing a
negative log likelihood (NLL) loss through stochastic gradient descent, much as language models are.

The internal transformer attention layers used here build on the
[GPT-Neo model](https://raw.githubusercontent.com/huggingface/transformers/e3cc4487fe66e03ec85970ea2db8e5fb34c455f4/src/transformers/models/gpt_neo/modeling_gpt_neo.py)
with minimal modifications only. However, the input layer, output layer, and arrangement of those internal
attention layers do differ from GPT-Neo and between the two pre-built model architectures. We detail these
next.

#### Conditionally Independent Point Process Transformer

For a conditionally independent model, the time and contents of the subsequent event at any point in the
sequence will be all predicted independently from one another given the representation of the sequence so far.
This is the most similar to a traditional transformer setting, and therefore the modifications are minor.

##### Input Layer

As this model does not need to separate out internal event covariates for dependent parsing, this input layer
simply embeds the contents of the event via a `DataEmbeddingLayer` into a single, fixed-size embedding, then
adds in temporal position emebeddings as well. This produces a single embedding of shape
`(batch_size, sequence_length, hidden_size)`, much like a traditional, NLP transformer would use.

##### Attention Layer

This model's internal transformer architecture is unchanged from the GPT-Neo model.

##### Output Layer

Here, the only difference from a traditional transformer is that rather than producing a single categorical
distribution over subsequent tokens, this model simultaneously emits a time-to-event distribution for the time
until the next event, categorical distributions for its categorical covariates, and continuous distributions
for its numerical covariates.

#### Nested Attention Point Process Transformer

A nested attention model must deviate further from a traditional transformer in order to account for the
intra-event dependencies specified in its configuration file.

##### Input Layer

To account for intra-event dependencies, the input layer for this model splits input event covariates into
dependency graph element groupings, via the specified configuration file. These groupings are each
independently embedded into fixed size embeddings, before being summed following the dependency graph
dimension (so that input embeddings depend on historical dependency graph elements as well). Thus, the output
here is no longer a traditional one-dimensional input, but is instead a two-dimensional attention input of
shape `(batch_size, sequence_length, dependency_graph_length, hidden_size)`.

##### Attention Layer

To reflect intra-event causal dependencies, this model breaks each attention layer down into three steps:

1. First, full-event embeddings are produced from input embeddings by taking the final element of the
   dependency graph as the "full event".
2. Next, history embeddings are produced by running sequential self-attention over the full event embedding
   sequence only.
3. Finally, internal event covariate embeddings are updated by running a second self-attention layer over
   the dependency graph elements, preceded by the prior event's history embedding, independently. The
   outputs of this layer is then used as the input to the subsequent layer.

This process can be seen pictorally here:
![Dependency-aware Attention](https://github.com/mmcdermott/EventStreamGPT/assets/470751/36e271f7-2101-4bfb-a4fb-ebe51d944e26)

##### Output Layer

Much like for the conditionally independent model, here we need to emit more complex distributions. However,
unlike that model, for the nested attention case these emissions are done in a per-dependency graph element
manner, such that each element of the dependency graph only emits predictions for the subsequent elements in
that graph. This means that the only cross-event predictions take the form of the prediction of the time until
the next event, which is produced off of the final element of the dependency graph in isolation.

#### Further documentation

For further details, see the documentation for the
[data embedding layer](https://eventstreamml.readthedocs.io/en/dev/api/EventStream.data.data_embedding_layer.html),
the various
[generative layers](https://eventstreamml.readthedocs.io/en/dev/api/EventStream.transformer.generative_layers.html)
supported, the specific model classes for the
[conditionally independent](https://eventstreamml.readthedocs.io/en/dev/api/EventStream.transformer.conditionally_independent_model.html)
and
[nested attention](https://eventstreamml.readthedocs.io/en/dev/api/EventStream.transformer.nested_attention_model.html)
models and input layers.
