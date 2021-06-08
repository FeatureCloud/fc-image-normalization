"""
    FeatureCloud Image Normalization Application

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
from .logic import AppLogic, bcolors
import numpy as np
import bios
import os


class CustomLogic(AppLogic):
    """ Subclassing AppLogic for overriding specific methods
        to implement the Image Normalization application.

    Attributes
    ----------
    train_filename: str
    test_filename: str
    stats: dict
        statistics of the dataset.
    method: str

    Methods
    -------
    read_config(config_file)
    read_input(path)
    local_preprocess(x_train, x_test, global_stats)
    global_aggregation(stats)
    write_results(train_set, test_set, output_path)
    """

    def __init__(self):
        super().__init__()
        self.train_filename = None
        self.test_filename = None
        self.stats = {}
        self.method = None  # For later use if other methods are implemented!

    def read_config(self, config_file):
        config = bios.read(config_file)['fc_image_normalization']
        self.train_filename = config['local_dataset']['train']
        self.test_filename = config['local_dataset']['test']
        self.method = config["method"]

        self.lazy_initialization(**config["logic"])

    def read_input(self, path):
        local_stats = {}
        train_path = path + "/" + self.train_filename
        print(f"{bcolors.VALUE}Reading {train_path} ...{bcolors.ENDC}")
        x_train, y_train, local_stats["train_mean"], local_stats["train_std"] = read_file(train_path)

        test_path = path + "/" + self.test_filename
        print(f"{bcolors.VALUE}Reading {test_path} ...{bcolors.ENDC}")
        # an empty numpy array with size == 0 is indicator
        # that there is no corresponding test file for a client
        x_test, y_test, local_stats["test_mean"], local_stats["test_std"] = read_file(test_path)
        local_stats["n_train_samples"] = x_train.shape[0]
        local_stats["n_test_samples"] = x_test.shape[0]

        return x_train, y_train, x_test, y_test, local_stats

    def local_preprocess(self, x_train, x_test, global_stats):
        if self.method == "variance":
            normalized_x_train = np.subtract(x_train, global_stats["train_mean"]) / global_stats["train_std"]
            if np.size(x_test) != 0:
                normalized_x_test = np.subtract(x_test, global_stats["test_mean"]) / global_stats["test_std"]
            else:
                normalized_x_test = np.array([])
        else:
            print(f"{bcolors.FAIL}{self.method} was not implemented as a normalization method.{bcolors.ENDC}")
        return normalized_x_train.tolist(), normalized_x_test.tolist()

    def global_aggregation(self, stats):
        if self.method == "variance":
            global_stats = {
                "train_mean": average([x["train_mean"] for x in stats], [x["n_train_samples"] for x in stats]),
                "train_std": average([x["train_std"] for x in stats], [x["n_train_samples"] for x in stats]),
                "test_mean": average([x["test_mean"] for x in stats], [x["n_test_samples"] for x in stats]),
                "test_std": average([x["test_std"] for x in stats], [x["n_test_samples"] for x in stats])}
        else:
            print(f"{bcolors.FAIL}{self.method} was not implemented as a normalization method.{bcolors.ENDC}")
        return global_stats

    def write_results(self, train_set, test_set, output_path):
        np.save(f"{output_path}/{self.train_filename}", train_set)
        if np.size(test_set) != 0:
            np.save(f"{output_path}/{self.test_filename}", test_set)


logic = CustomLogic()


def read_file(path):
    """ load a numpy file and compute mean and std. for each channel

    Parameters
    ----------
    path: str

    Returns
    -------

    """
    if os.path.exists(path):
        x, y = np.load(path, allow_pickle=True)
        x = np.array(list(x))
        sample_shape = x[0].shape[1:]
        if min(sample_shape) > 3:
            x = np.expand_dims(x, axis=-1)
        mean = np.mean(x, axis=tuple(range(x.ndim - 1)))
        std = np.std(x, axis=tuple(range(x.ndim - 1)))
        print(f"{bcolors.VALUE} {path} was read successfully{bcolors.ENDC}")
        return x, y.tolist(), mean, std
    else:
        print(f"{bcolors.WARNING} {path} File Not Found!!!{bcolors.ENDC}")
        return np.array([]), [], None, None


def average(stats, n_samples):
    """ Weighted averages of statistics using number of samples

    Parameters
    ----------
    stats: list
    n_samples: list

    Returns
    -------

    """
    print(stats)
    total = np.sum(n_samples)
    avg = None
    for client_ind, client_stat in enumerate(stats):
        if n_samples[client_ind] > 0:
            if avg is None:
                avg = np.zeros(client_stat.ndim)
            for channel_ind in range(client_stat.ndim):
                avg[channel_ind] += client_stat[channel_ind] * n_samples[client_ind]
    avg /= total
    return avg
