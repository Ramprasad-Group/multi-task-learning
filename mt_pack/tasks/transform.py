"""Hold preprocessing functions for the data set"""
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
from typing import List, Dict
import numpy as np
import tensorflow as tf
from pathlib import Path

def kfold_split(df, conf):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    kf = KFold(n_splits=conf['kfolds'], shuffle=False, random_state=0)
    datasets = []
    for idx_train, idx_val in kf.split(df):
        train, val = df.iloc[idx_train].copy(), df.iloc[idx_val].copy()
        datasets.append(dict(train=train, val=val))
    return datasets


def scale(datasets):
    for num, data in enumerate(datasets):
        # fps
        datasets[num]['fp_scaler'] = MinMaxScaler()
        fps_names = data['train'].loc[:, data['train'].columns.str.startswith('fingerprints')].columns.values.tolist()
        datasets[num]['train'].loc[:, fps_names] = datasets[num]['fp_scaler'].fit_transform(data['train'][fps_names])
        datasets[num]['val'].loc[:, fps_names] = datasets[num]['fp_scaler'].transform(data['val'][fps_names])

        # props
        datasets[num]['prop_scaler'] = RobustScaler()
        props_name = data['train'].columns[~data['train'].columns.str.startswith('fingerprints')].values.tolist()

        datasets[num]['train'].loc[:, props_name] = datasets[num]['prop_scaler'].fit_transform(data['train'][props_name])
        datasets[num]['val'].loc[:, props_name] = datasets[num]['prop_scaler'].transform(data['val'][props_name])

    return datasets

def prepare_and_read_dataframe(conf: dict):
    df = pd.read_pickle(conf['dataset'])

    # Change names, remove 'properties.'
    map_dict = {}
    for prop, prop_dict in conf['properties'].items():
        map_dict[f"properties.{prop}"] = f"{prop_dict['name']}"
    df = df.rename(columns=map_dict)

    return df

def get_dataset_mt2(conf: dict) -> List[Dict]:

    df = prepare_and_read_dataframe(conf)
    fps = df.loc[:, df.columns.str.startswith('fingerprints')].columns.values.tolist()

    props = [v['name'] for k, v in conf['properties'].items()]
    df = df[props + fps]


    # Takes time
    df = df.melt(id_vars=fps).dropna().reset_index(drop=True)
    dum = pd.get_dummies(df['variable'])[props]
    df = pd.concat([dum, df], 1, keys=['dummy','fps'])

    new_index = df.columns.to_list()
    new_index[-2:] = [('data', 'variable'), ('data', 'value')]
    df.columns = pd.MultiIndex.from_tuples(new_index)
    
    # # Split
    #

    kf = StratifiedKFold(n_splits=conf['kfolds'], shuffle=True, random_state=20)
    datasets = []
    for idx_train, idx_val in kf.split(df, df.data.variable):
        train, val = df.iloc[idx_train].copy(), df.iloc[idx_val].copy()
        datasets.append(dict(train=train, val=val))
    
    # # Scale 
    # 
    for num, data in enumerate(datasets):
        # fps
        datasets[num]['fp_scaler'] = MinMaxScaler()
        datasets[num]['train'].fps = datasets[num]['fp_scaler'].fit_transform(data['train'].fps)
        datasets[num]['val'].fps = datasets[num]['fp_scaler'].transform(data['val'].fps)

        # scale property values
        sc = RobustScaler

        datasets[num]['prop_scaler'] = {}

        for prop in datasets[num]['train'].dummy.columns.to_list():
            _sc = sc()

            # Train
            cond = datasets[num]['train'].data.variable == prop
            datasets[num]['train'].loc[cond, ('data', ['value'])]  = _sc.fit_transform(datasets[num]['train'].loc[cond, ('data', ['value'])] )

            # val
            cond = datasets[num]['val'].data.variable == prop
            datasets[num]['val'].loc[cond, ('data', ['value'])]  = _sc.transform(datasets[num]['val'].loc[cond, ('data', ['value'])] )

            datasets[num]['prop_scaler'][prop] = _sc
    
    for num, dataset in enumerate(datasets):
        for set_name in ['train', 'val']:
            # fps
            fps = dataset[set_name].fps.astype('float32')
            value = dataset[set_name].data.value.astype('float32')
            selector = dataset[set_name].dummy.astype('float32')
            var = dataset[set_name].data.variable

            data = tf.data.Dataset.from_tensor_slices(
                ((fps.values, selector.values, var.values), value.values)
            )

            data = data.cache().batch(conf['batchsize']).prefetch(tf.data.experimental.AUTOTUNE)
            datasets[num][set_name] = data

    return datasets

def get_dataset_mt(conf: dict) -> List[Dict]:
    df = prepare_and_read_dataframe(conf)
    fps = df.loc[:, df.columns.str.startswith('fingerprints')].columns.values.tolist()

    props = [v['name'] for k, v in conf['properties'].items()]
    df = df[props + fps] 

    # process
    datasets = kfold_split(df, conf)
    datasets = scale(datasets)
    return datasets

def get_dataset_st(conf: dict, prop_name: str) -> List[Dict]:
    df = prepare_and_read_dataframe(conf)

    # select and cut
    fps = df.loc[:, df.columns.str.startswith('fingerprints')].columns.values.tolist()
    df = df[[prop_name] + fps] 
    df.dropna(subset=[prop_name], inplace=True)

    # process
    datasets = kfold_split(df, conf)
    datasets = scale(datasets)

    return datasets

def to_tf_datasets(
    datasets: dict, conf: dict) -> dict:

    for num, dataset in enumerate(datasets):

        for set_name in ['train', 'val']:
            dataset[set_name] = dataset[set_name].astype('float32')

            # train
            fps_names = dataset[set_name].loc[:, dataset[set_name].columns.str.startswith('fingerprints')].columns.values.tolist()
            prop_names = dataset[set_name].loc[:, ~dataset[set_name].columns.str.startswith('fingerprints')].columns.values.tolist()

            fingerprints = dataset[set_name][fps_names]
            properties = dataset[set_name][prop_names]

            data = tf.data.Dataset.from_tensor_slices(
                (fingerprints.values, properties.to_dict("list"))
            )
            data = data.cache().batch(conf['batchsize']).prefetch(tf.data.experimental.AUTOTUNE)
            datasets[num][set_name] = data

    return datasets