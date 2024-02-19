"""Trains the NN-MT models"""
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

from mt_pack.tasks.transform import get_dataset_mt, to_tf_datasets
from mt_pack.utils import get_checkpoint_log
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel

@click.group(help="MT NN")
@click.pass_context
def nn_mt(ctx):
    logging.info("NN MT models")
    conf = ctx.obj


@nn_mt.command(name="train", help="Train model")
@click.pass_context
def train(ctx):
    conf = ctx.obj
    props = [v['name'] for k, v in conf['properties'].items()]


    datasets = get_dataset_mt(conf)
    datasets = to_tf_datasets(datasets, conf)

    dds = []
    for fold, dataset in enumerate(datasets):
        logging.info(f"NN-MT: {props}, fold: {fold}")
        ckpt_path, logdir = get_checkpoint_log(conf, f"multi_task/{fold}/")

        tuner = search_hps(conf, dataset)
        model = tuner.get_best_models(num_models=1)[0]

        # predictions
        sc = dataset['prop_scaler'].inverse_transform
        pred = model.predict(dataset['val'])
        pred_df = sc(pd.DataFrame(pred))
        
        truth_df = pd.concat([pd.DataFrame(i[1]) for i in dataset['val'].as_numpy_iterator()], ignore_index=True)
        truth_df: pd.DataFrame = sc(truth_df)

        for name, truth_col in truth_df.iteritems():
            # import ipdb; ipdb.set_trace()
            nans = truth_col.isnull()
            pred = pred_df[name][~nans]
            truth = truth_col[~nans]

            _d = dict(
                rmse=mean_squared_error(pred, truth, squared=False),
                r2=r2_score(pred, truth),
                me=max_error(pred, truth),
                mae=mean_absolute_error(pred, truth),
                evs=explained_variance_score(pred, truth),
                prop=name,
                task="NN-MT",
                type='val',
                fold=fold
            )

            print(_d)
            dds.append(_d)

        Path(ckpt_path).parent.joinpath('fp_scaler.pkl').write_bytes(pickle.dumps(dataset['fp_scaler']))
        Path(ckpt_path).parent.joinpath('prop_scaler.pkl').write_bytes(pickle.dumps(dataset['prop_scaler']))

    df = pd.DataFrame(dds)
    df.to_csv(conf["results"].joinpath(f"k_{conf['kfolds']}_nn_mt.csv"))
    print(df)



##### Loss function
def remove_nan(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    nans = tf.math.is_nan(y_true)
    zero = tf.constant(0.0, dtype=y_pred.dtype)

    y_true = tf.where(nans, zero, y_true)
    y_pred = tf.where(nans, zero, y_pred)
    return y_true, y_pred

def loss_mse(y_true, y_pred, sample_weight=None):
    y_true, y_pred = remove_nan(y_true, y_pred)
    return tf.keras.losses.mse(y_pred, y_true)

################
#### ###
## HyperParameter Search

def search_hps(conf, dataset):

    tuner = Hyperband(
        MyHyperModel(conf),
        objective='val_loss',
        max_epochs=100,
        seed=10,
        directory='hyperparamter_search',
        project_name='nn-mt')

    print(tuner.search_space_summary())

    tuner.search(dataset['train'],
                epochs=100,
                validation_data=dataset['val'])

    print(tuner.results_summary())
    return tuner

class MyHyperModel(HyperModel):

    def __init__(self, conf):
        self.conf = conf

    def build(self, hp):

        model = NNMT_HP(self.conf, hp)
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss=loss_mse)

        return model

class NNMT_HP(keras.Model):
    def __init__(self, conf, hp):
        super().__init__()
        self.properties = [v['name'] for k, v in conf['properties'].items()]
        self.my_layers = []
        for i in range(hp.Int('num_layers', 1, 2)):
            self.my_layers.append(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),))
            
            self.my_layers.append(tf.keras.layers.ReLU())
            self.my_layers.append(tf.keras.layers.Dropout(0.5))

        self.last_layer = tf.keras.layers.Dense(len(self.properties))


    def call(self, inputs):
        out = inputs
        for layer in self.my_layers:
            out = layer(out)
        out = self.last_layer(out)

        return_dict = {}
        for num, prop in enumerate(self.properties):
            return_dict[prop] = out[..., num] 
        return return_dict
