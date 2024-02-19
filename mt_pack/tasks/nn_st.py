"""Trains the NN-ST models"""
import logging
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error, r2_score)
from tensorflow import keras

from mt_pack.tasks.transform import get_dataset_st, to_tf_datasets
from mt_pack.utils import get_checkpoint_log

import matplotlib.pyplot as plt
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel


@click.group(help="ST NN")
@click.pass_context
def nn_st(ctx):
    logging.info("NN ST models")
    conf = ctx.obj

@nn_st.command(name="train", help="Train model")
@click.pass_context
def train(ctx):
    conf = ctx.obj

    dds = []
    for prop, prop_dict in conf["properties"].items():        
        datasets = get_dataset_st(conf, prop_dict['name'])
        datasets = to_tf_datasets(datasets, conf)

        for fold, dataset in enumerate(datasets):
            logging.info(f"NN-ST: {prop_dict['name']}, fold: {fold}")

            ckpt_path, logdir = get_checkpoint_log(conf, f"single_task/{prop_dict['name']}/{fold}/")

            # Hyperparamter search
            tuner = search_hps(prop_dict, dataset)
            model = tuner.get_best_models(num_models=1)[0]

            # predictions
            sc = dataset['prop_scaler'].inverse_transform
            pred = model.predict(dataset['val'])[prop_dict['name']]           
            pred = sc(pred.flatten().reshape(-1, 1))

            truth = [i[1][prop_dict['name']] for i in dataset['val'].as_numpy_iterator()]
            truth = np.concatenate(truth, axis=0)
            truth = sc(truth.flatten().reshape(-1, 1))

            _d = dict(
                rmse=mean_squared_error(pred, truth, squared=False),
                r2=r2_score(pred, truth),
                me=max_error(pred, truth),
                mae=mean_absolute_error(pred, truth),
                evs=explained_variance_score(pred, truth),
                prop=prop_dict['name'],
                task="NN-ST",
                type='val',
                fold=fold
            )

            print(_d)

            # save scalers            
            Path(ckpt_path).parent.joinpath('fp_scaler.pkl').write_bytes(pickle.dumps(dataset['fp_scaler']))
            Path(ckpt_path).parent.joinpath('prop_scaler.pkl').write_bytes(pickle.dumps(dataset['prop_scaler']))
            dds.append(_d)
            
    df = pd.DataFrame(dds)
    df.to_csv(conf["results"].joinpath(f"k_{conf['kfolds']}_nn_st.csv"))
    print(df)


################
#### ###
## HyperParameter Search

def search_hps(prop_dict, dataset):
    
    tuner = Hyperband(
        MyHyperModel(prop_dict['name']),
        objective='val_loss',
        max_epochs=100,
        seed=10,
        directory='hyperparamter_search',
        project_name='nn-st')

    print(tuner.search_space_summary())

    tuner.search(dataset['train'],
                epochs=100,
                validation_data=dataset['val'])

    print(tuner.results_summary())
    return tuner

class MyHyperModel(HyperModel):

    def __init__(self, prop):
        self.prop = prop

    def build(self, hp):

        model = NNST_HP(self.prop, hp)
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='mse')

        return model

class NNST_HP(keras.Model):
    def __init__(self, prop_name, hp):
        super().__init__()
        self.prop = prop_name
        self.my_layers = []
        for i in range(hp.Int('num_layers', 1, 2)):
            self.my_layers.append(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),))
            
            self.my_layers.append(tf.keras.layers.ReLU())
            self.my_layers.append(tf.keras.layers.Dropout(0.5))

        self.last_layer = tf.keras.layers.Dense(1, name=prop_name)

    def call(self, inputs):
        out = inputs
        for layer in self.my_layers:
            out = layer(out)
        return {self.prop: self.last_layer(out)}
