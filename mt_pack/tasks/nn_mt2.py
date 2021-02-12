"""Trains the NN-MT2 models"""
import logging
import pickle
from pathlib import Path
from typing import Dict

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband, RandomSearch
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error, r2_score)
from tensorflow import keras
from tensorflow.keras import callbacks
import tensorflow_addons as tfa

from mt_pack.task_new.transform import get_dataset_mt2, to_tf_datasets
from mt_pack.utils import get_checkpoint_log
from sklearn.preprocessing import MinMaxScaler, RobustScaler

np.random.seed(10)
tf.random.set_seed(10)

@click.group(help="MT2 NN")
@click.pass_context
def nn_mt2(ctx):
    logging.info("NN MT2 models")

@nn_mt2.command(name="train", help="Train model")
@click.pass_context
def train(ctx):
    conf = ctx.obj
    props = [v['name'] for k, v in conf['properties'].items()]

    datasets = get_dataset_mt2(conf)

    dds = []
    for fold, dataset in enumerate(datasets):

        logging.info(f"NN-MT2: {props}, fold: {fold}")
        ckpt_path, logdir = get_checkpoint_log(conf, f"multi_task_mt2/{fold}/")

        tuner = search_hps(dataset, fold)
        model = tuner.get_best_models(num_models=1)[0]

     
        # predictions
        pred = model.predict(dataset['val'])
        prop_names = np.concatenate([i[0][2] for i in dataset['val'].as_numpy_iterator()], 0).flatten().astype(np.str)
        pred_df = pd.DataFrame(np.concatenate([pred, prop_names[:,np.newaxis]], 1), columns=['value', 'variable'])
        pred_df.value = pred_df.value.astype('float32')

        truth_df = pd.concat([pd.DataFrame(i[1]) for i in dataset['val'].as_numpy_iterator()], ignore_index=True)
        truth_df = pd.DataFrame(np.concatenate([truth_df, prop_names[:,np.newaxis]], 1), columns=['value', 'variable'])
        truth_df.value = truth_df.value.astype('float32')

        props = pred_df.variable.unique().tolist()
        for prop in props:
            sc = dataset['prop_scaler'][prop].inverse_transform
            cond = pred_df.variable == prop
            pred_df.loc[cond, ['value']] = sc(pred_df.loc[cond, ['value']].values)
            truth_df.loc[cond, ['value']] = sc(truth_df.loc[cond, ['value']].values)

            pred = pred_df.loc[cond, ['value']]
            truth = truth_df.loc[cond, ['value']]

            _d = dict(
                rmse=mean_squared_error(pred, truth, squared=False),
                r2=r2_score(pred, truth),
                me=max_error(pred, truth),
                mae=mean_absolute_error(pred, truth),
                evs=explained_variance_score(pred, truth),
                prop=prop,
                task="NN-MT2",
                type='val',
                fold=fold
            )

            print(_d)
            dds.append(_d)

        Path(ckpt_path).parent.joinpath('fp_scaler.pkl').write_bytes(pickle.dumps(dataset['fp_scaler']))
        Path(ckpt_path).parent.joinpath('prop_scaler.pkl').write_bytes(pickle.dumps(dataset['prop_scaler']))
    
    df = pd.DataFrame(dds)
    df.to_csv(conf["results"].joinpath(f"k_{conf['kfolds']}_nn_mt2.csv"))
    print(df)




################
#### ###
## HyperParameter Search

def search_hps(dataset, fold, sp = None):

    tuner = Hyperband(
        MyHyperModel(scaler_path=sp if sp else None),
        objective='val_loss',
        max_epochs=400,
        seed=10,
        directory='hyperparamter_search',
        project_name='nn-mt2_' + str(fold),
        )

    print(tuner.search_space_summary())

    earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200)


    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.9,
        patience=20,
        cooldown=5,
        verbose=True,
    )

    tuner.search(dataset['train'],
                epochs=100,
                validation_data=dataset['val'],
                callbacks=[earlystop, reduce_lr])

    # print(tuner.results_summary(num_trials=1))
    return tuner

class MyHyperModel(HyperModel):

    def __init__(self, scaler_path=None):
        if scaler_path:
            self.scaler_path = scaler_path
        else:
            self.scaler_path = None
            

    def build(self, hp):

        model = NNMT2_HP(hp, scaler_path=self.scaler_path if self.scaler_path else None)
        opt = keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3]))
        
        stocastic_avg_sgd = tfa.optimizers.SWA(opt)
        # loss_weights = {'sol': 100, 'Eat': 100}

        model.compile(
            optimizer=stocastic_avg_sgd,
            loss='mse')

        return model

class NNMT2_HP(keras.Model):
    def __init__(self, hp, scaler_path=None):
        super().__init__()

        if scaler_path:
            self.fp_scaler: MinMaxScaler = pickle.loads(scaler_path.joinpath('fp_scaler.pkl').read_bytes())
            self.prop_scalers: Dict[RobustScaler] = pickle.loads(scaler_path.joinpath('prop_scaler.pkl').read_bytes())

        self.my_layers = []
        # self.bn1 = tf.keras.layers.BatchNormalization()
        for i in range(hp.Int('num_layers', 2, 2)):
            self.my_layers.append(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=128,
                                            max_value=512,
                                            step=32),))
            
            self.my_layers.append(tf.keras.layers.ReLU())
            self.my_layers.append(tf.keras.layers.Dropout(hp.Float(
                'dropout_' + str(i),
                min_value=0.0,
                max_value=0.7,
                default=0.25,
                step=0.1,
            )))

        self.last_layer = tf.keras.layers.Dense(1)


    def call(self, inputs):
        out = tf.concat(inputs[:2], 1)
        # out = self.bn1(out)
        for layer in self.my_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out
    

