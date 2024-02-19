"""Trains the GRP-ST models"""
import logging
import pickle

import click
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error, r2_score)

from pathlib import Path
from mt_pack.tasks.transform import get_dataset_st

@click.group(help="Train GPR models")
@click.pass_context
def gpr(ctx):
    logging.info("GPR models")
    conf = ctx.obj

@gpr.command(name="train", help="Train model")
@click.pass_context
def train(ctx):
    conf = ctx.obj

    dds = []
    for prop, prop_dict in conf["properties"].items():
        datasets = get_dataset_st(conf, prop_dict['name'])
        for fold, dataset in enumerate(datasets):
            logging.info(f"GPR: {prop_dict['name']}, fold: {fold}")

            # init GPR
            kernel = RBF(length_scale=1) + WhiteKernel(1e-4)
            gpr = GaussianProcessRegressor(kernel=kernel)

            gpr.fit(dataset["train"].drop(columns=prop_dict['name']), dataset["train"][prop_dict['name']])

            # Inverse scaling
            sc = dataset['prop_scaler'].inverse_transform
            pred = gpr.predict(dataset["val"].drop(columns=prop_dict['name']))
            pred = sc(pred.reshape(-1, 1))
            truth = sc(dataset["val"][[prop_dict['name']]])

            _d = dict(
                rmse=mean_squared_error(pred, truth, squared=False),
                r2=r2_score(pred, truth),
                me=max_error(pred, truth),
                mae=mean_absolute_error(pred, truth),
                evs=explained_variance_score(pred, truth),
                prop=prop_dict['name'],
                task="GPR-ST",
                type='val',
                fold=fold
            )

            print(_d)

            # save model 
            fl : Path = conf["checkpoint_path"].joinpath(f"gpr/{prop_dict['name']}/{fold}")
            fl.mkdir(exist_ok=True, parents=True)
            fl.joinpath('gpr.pkl').write_bytes(pickle.dumps(gpr))
            # save scalers
            fl.joinpath('fp_scaler.pkl').write_bytes(pickle.dumps(dataset['fp_scaler']))
            fl.joinpath('prop_scaler.pkl').write_bytes(pickle.dumps(dataset['prop_scaler']))

            dds.append(_d)

    df = pd.DataFrame(dds)
    df.to_csv(conf["results"].joinpath(f"k_{conf['kfolds']}_gpr.csv"))
    print(df)

