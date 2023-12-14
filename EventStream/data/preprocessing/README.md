# Polars friendly pre-processing models.

A collection of pre-processing (outlier detection and normalization) models that can be fit via polars
expressions, either directly on a dataframe or in a group-by context. All only work with univariate data at
present.

## StandardScaler

Computes the mean and standard deviation of the data. Upon predict, subtracts the mean and divides by the
standard deviation.

## StddevCutoff

Removes all values that occur more than a specified threshold of standard deviations away from the mean.
