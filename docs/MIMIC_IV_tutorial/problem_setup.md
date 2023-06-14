## Tutorial Problem Set-up

In this user guide, we will follow the steps required to process data and produce foundation models over a
small cohort from the MIMIC-IV dataset {cite:p}`mimiciv`

MIMIC-IV is a publicly available dataset consisting of the EHR data for all adult patients who were admitted
to the emergency department (ED) or an intensive care unit (ICU) at Beth Israel Deaconess Medical Center
(BIDMC) between 2008 and 2019. This dataset contains approximately 300,000 patients and consists of numerous
modalities of health data, including diagnoses, laboratory test results, medications, in and out of hospital
mortality, and many others, all localized continuously in time over all admissions of a single patient to the
BIDMC.

Our task with these data is to build a generative model over the continuous-time, complex event stream data
contained in MIMIC-IV. This can also be seen as a multi-variate marked temporal point process. In particular,
given a sequence of complex events
$\vec x_1, \ldots, \vec x_N$ which occur at continuous times $t_1, \ldots, t_N$, we wish to produce a
model of the following probability distribution:

$$
p(t_i, \vec x_i | \underbrace{(t_1, \vec x_1), \ldots, (t_{i-1}, \vec x_{i-1})}_{\vec h_{i-1}})
$$

We will realize this through a transformer neural network architecture parametrized by $\vec \theta$, such
that $f_{\vec \theta} (t_i, \vec x_i, \vec h_{i-1}) = p(t_i, \vec x_i | \vec h_{i-1})$. Note that here it may
be the case that internal covariates of each event $\vec x_i$ have internal causal dependencies. For example,
if $\vec x_i^{(j)}$ is used to denote the $j$th internal covariate of event $i$, then $p(\vec x_i | \vec
h_{i-1}, t_i) \neq \prod_{j} p(\vec x_i^{(j)} | \vec h_{i-1}, t_i)$. Any full generative model will
therefore need to account for these internal causal dependencies.
