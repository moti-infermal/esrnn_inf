[![Build](https://github.com/kdgutier/esrnn_torch/workflows/Python%20package/badge.svg?branch=master)](https://github.com/kdgutier/esrnn_torch/tree/master)
[![PyPI version fury.io](https://badge.fury.io/py/ESRNN.svg)](https://pypi.python.org/pypi/ESRNN/)
[![Downloads](https://pepy.tech/badge/esrnn)](https://pepy.tech/project/esrnn)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/kdgutier/esrnn_torch/blob/master/LICENSE)


# Fork of the Pytorch Implementation of the ES-RNN
Pytorch implementation of the ES-RNN algorithm proposed by Smyl, winning submission of the M4 Forecasting Competition. The class wraps fit and predict methods to facilitate interaction with Machine Learning pipelines along with evaluation and data wrangling utility. Developed by [Autonlab](https://www.autonlab.org/)’s members at Carnegie Mellon University.

## Installation Prerequisites
* numpy>=1.16.1
* pandas>=0.25.2
* pytorch>=1.3.1

## Installation

You can install the *released version* of `ESRNN` from the [Python package index](https://pypi.org) with:

```python
pip install ESRNN-INF
```

## Usage

### Input data

The fit method receives `X_df`, `y_df` training pandas dataframes in long format. Optionally `X_test_df` and `y_test_df` to compute out of sample performance.
- `X_df` must contain the columns `['unique_id', 'ds', 'x']`
- `y_df` must contain the columns `['unique_id', 'ds', 'y']`
- `X_test_df` must contain the columns `['unique_id', 'ds', 'x']`
- `y_test_df` must contain the columns `['unique_id', 'ds', 'y']` and a benchmark model to compare against  (default `'y_hat_naive2'`).

For all the above:
- The column `'unique_id'` is a time series identifier, the column `'ds'` stands for the datetime.
- Column `'x'` is an exogenous categorical feature.
- Column `'y'` is the target variable.
- Column `'y'` **does not allow negative values** and the first entry for all series must be **grater than 0**.

The `X` and `y` dataframes must contain the same values for `'unique_id'`, `'ds'` columns and be **balanced**, ie.no *gaps* between dates for the frequency.


<center>

|`X_df`|`y_df` |`X_test_df`| `y_test_df`|
|:-----------:|:-----------:|:-----------:|:-----------:|
|<img src="https://raw.githubusercontent.com/kdgutier/esrnn_torch/master/.github/images/x_train.png" width="152"> | <img src="https://raw.githubusercontent.com/kdgutier/esrnn_torch/master/.github/images/y_train.png" width="152"> | <img src="https://raw.githubusercontent.com/kdgutier/esrnn_torch/master/.github/images/x_test.png" width="152"> | <img src="https://raw.githubusercontent.com/kdgutier/esrnn_torch/master/.github/images/y_test.png" width="223">|

</center>


## Overall Weighted Average

A metric that is useful for quantifying the aggregate error of a specific model for various time series is the Overall Weighted Average (OWA) proposed for the M4 competition. This metric is calculated by obtaining the average of the symmetric mean absolute percentage error (sMAPE) and the mean absolute scaled error (MASE) for all the time series of the model and also calculating it for the Naive2 predictions. Both sMAPE and MASE are scale independent. These measurements are calculated as follows:

![OWA](https://raw.githubusercontent.com/kdgutier/esrnn_torch/master/.github/images/metrics.png)



## Current Results
Here we used the model directly to compare to the original implementation. It is worth noticing that these results do not include the ensemble methods mentioned in the [ESRNN paper](https://www.sciencedirect.com/science/article/pii/S0169207019301153).<br/>
[Results of the M4 competition](https://www.researchgate.net/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward).
<br/>

| DATASET   | OUR OWA | M4 OWA (Smyl) |
|-----------|:---------:|:--------:|
| Yearly    | 0.785   | 0.778  |
| Quarterly | 0.879   | 0.847  |
| Monthly   | 0.872   | 0.836  |
| Hourly    | 0.615   | 0.920  |
| Weekly    | 0.952   | 0.920  |
| Daily     | 0.968   | 0.920  |


## Replicating M4 results


Replicating the M4 results is as easy as running the following line of code (for each frequency) after installing the package via pip:

```console
python -m ESRNN.m4_run --dataset 'Yearly' --results_directory '/some/path' \
                       --gpu_id 0 --use_cpu 0
```

Use `--help` to get the description of each argument:

```console
python -m ESRNN.m4_run --help
```

## Authors
This repository was developed with joint efforts from AutonLab researchers at Carnegie Mellon University and Orax data scientists.
* **Kin Gutierrez** - [kdgutier](https://github.com/kdgutier)
* **Cristian Challu** - [cristianchallu](https://github.com/cristianchallu)
* **Federico Garza** - [FedericoGarza](https://github.com/FedericoGarza) - [mail](fede.garza.ramirez@gmail.com)
* **Max Mergenthaler** - [mergenthaler](https://github.com/mergenthaler)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/kdgutier/esrnn_torch/blob/master/LICENSE) file for details.


## REFERENCES
1. [A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207019301153)
2. [The M4 Competition: Results, findings, conclusion and way forward](https://www.researchgate.net/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward)
3. [M4 Competition Data](https://github.com/M4Competition/M4-methods/tree/master/Dataset)
4. [Dilated Recurrent Neural Networks](https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf)
5. [Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition](https://arxiv.org/abs/1701.03360)
6. [A Dual-Stage Attention-Based recurrent neural network for time series prediction](https://arxiv.org/abs/1704.02971)
