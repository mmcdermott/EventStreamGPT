# Variance Impact Outlier Detector

## Description
The goal of this module is to provide a system that can quickly identify datapoints that have an over-sized
impact on sample variance to flag them as potential outliers. This definition of "outlier" is motivated by
machine learning applications. In such applications, data are often normalized prior to being fed into a
training model. If data are normalized on a training dataset which contains a point that is far more extreme
(having a greater distance from the mean) than other points in the dataset, it will have a proportionally
large impact on the train-set sample variance while having only a low probability of being observed in
validation or test sets. Thus, not accounting for these points when computing normalization statistics will
result in strategies that yield validation and held-out set feature distributions that do not have unit
variance.
