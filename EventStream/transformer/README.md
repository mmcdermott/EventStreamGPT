# Event-stream Transformer

Contains utilities for building transformer (in particular generative point-processes) models over
`EventStreamDataset`s.

TODO(mmd): Update documentation here.

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

A pictoral representation:
![Dependency-aware Attention](https://user-images.githubusercontent.com/470751/217272929-0b972d7f-793a-46f8-ac01-74d428bd7fcb.png)

### Output Layer

The output layer produces categorical distribution estimates for categorical variables (via logit scores per
vocabulary element) and numerical pytorch distribution objects for continuous variables, including
time-to-next-event. Currently, time-to-next-event can either be modeled via an exponential or a mixture of
lognormal distributions, and other continuous variables can only be modeled via output gaussian distributions.
