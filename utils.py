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
from utils import load_numpy
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
    data = load_numpy(filename)
    if target == "same-sep":
        x, y = data
        x = np.array(list(x))
    elif target == 'same-last':
        x = [s[-1] for s in data]
        y = [s[-1] for s in data]
    elif '.npy' in target or '.npz' in target:
        x = data
        y = np.load(target, allow_pickle=True)
    x = np.array(list(x))
    sample_shape = x[0].shape[1:]
    if min(sample_shape) > 3:
        x = np.expand_dims(x, axis=-1)
    mean = np.mean(x, axis=tuple(range(x.ndim - 1)))
    std = np.std(x, axis=tuple(range(x.ndim - 1)))
    return x, y.tolist(), mean, std
