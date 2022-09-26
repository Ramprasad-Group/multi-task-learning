# Polymer Informatics with Multi-Task Learning 

This repository contains the code to the [paper](https://www.cell.com/patterns/fulltext/S2666-3899(21)00058-1). The code trains four different machine learning models for the prediction of polymer properties. Two of the models are single-task (nn_st.py and gpr.py) and two are multi-task (nn_mt2.py and nn_mt.py) models. Please see the paper for more details.

## Prerequisites

- [Poetry](https://python-poetry.org/docs/) must be installed. See https://python-poetry.org/docs/#installation

## Install

1. Clone repo 
```bash
git clone https://gitlab.com/ramprasad-group/multi-task-learning && cd multi-task-learning
```

2. Init poetry
```bash
poetry install
poetry shell
cd .. && mkdir test && cd test
```

3. Run

```bash 
mtask -h
```

## Use

1. In the test directory

```bash
mkdir dataset && cd dataset
```

2. Place dataset in the `dataset/` directory

3. Run

```bash
mtask_new nn-st train
or
mtask_new nn-mt train
or
mtask_new nn-st2 train
```


