"""
    FeatureCloud Cross Validation Application
    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import numpy as np
from FeatureCloud.app.engine.app import LogLevel, app, SMPCOperation
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server

from bottle import Bottle


def run(host='localhost', port=5000):
    """ run the docker container on specific host and port.

    Parameters
    ----------
    host: str
    port: int

    """

    app.register()
    server = Bottle()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host=host, port=port)


def load_numpy(file_name):
    ds = np.load(file_name, allow_pickle=True)
    format = file_name.strip().split(".")[1].lower()
    if format == "npz":
        return ds['arr_0']
    return ds


def read_file(filename, target):
    """ load a numpy file and compute mean and std. for each channel
    Parameters
    ----------
    filename: str
    target: str
    Returns
    -------
    """
    format = filename.strip().split(".")[-1].strip()
    if format == 'npz':
        ds = np.load(filename)
        data, targets = ds['data'], ds['targets']
    else:
        np_file = load_numpy(filename)
        if target == "same-sep":
            data, targets = np_file
            data = np.array(list(data))
        elif target == 'same-last':
            data = [s[-1] for s in np_file]
            targets = np.array([s[-1] for s in np_file])
        elif '.npy' in target or '.npz' in target:
            data = np_file
            targets = np.load(target, allow_pickle=True)
    sample_shape = data[0].shape[1:]
    if min(sample_shape) > 3:
        x = np.expand_dims(data, axis=-1)
    mean = np.mean(x, axis=tuple(range(x.ndim - 1)))
    std = np.std(x, axis=tuple(range(x.ndim - 1)))
    return data, targets.tolist(), mean, std


def save_numpy(file_name, features, labels, target):
    format = file_name.strip().split(".")[1].lower()
    save = {"npy": np.save, "npz": np.savez_compressed}
    if target == "same-sep":
        save[format](file_name, np.array([features, labels]))
    elif target == "same-last":
        samples = [np.append(features[i], labels[i]) for i in range(features.shape[0])]
        save[format](file_name, samples)
    elif target.strip().split(".")[1].lower() == 'npy':
        np.save(file_name, features)
        np.save(target, labels)
    elif target.strip().split(".")[1].lower() in 'npz':
        np.savez_compressed(file_name, features)
        np.savez_compressed(target, labels)
    else:
        return None


def to_np(data):
    if type(data) is list:
        return np.array([to_np(item) for item in data])
    return data
