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
import copy
import os.path
from FeatureCloud.engine.app import app_state, AppState, Role, LogLevel, SMPCOperation
from FeatureCloud.engine.app import State as op_state
import numpy as np
from utils import save_numpy
from .utils import read_file
import ConfigState

name = 'image_normalization'


@app_state(name='initial', role=Role.BOTH, app_name=name)
class LocalStats(ConfigState.State):

    def register(self):
        self.register_transition('WriteResults', Role.PARTICIPANT)
        self.register_transition('GlobalStats', Role.COORDINATOR)

    def run(self):
        self.lazy_init()
        self.read_config()
        self.finalize_config()
        self.update(progress=0.1)
        self.store('method', self.config['method'])
        self.store('target_value', self.config['local_dataset']['target_value'])
        stats = self.read_files()
        self.store('smpc_used', self.config.get('use_smpc', False))
        self.send_data_to_coordinator(data=stats, use_smpc=self.load('smpc_used'))
        self.update(progress=0.3)
        if self.is_coordinator:
            return 'GlobalStats'
        return 'WriteResults'

    def read_files(self):
        x_train, y_train, x_test, y_test, local_stats = [], [], [], [], []
        splits = zip(self.load('input_files')['train'], self.load('input_files')['test'])
        for split_train_file, split_test_file in splits:
            if not os.path.isfile(split_train_file):
                self.log(f"File not found:\n{split_train_file}", LogLevel.ERROR)
                self.update(state=op_state.ERROR)
            x, y, mean_train, std_train = read_file(split_train_file,
                                                                self.config['local_dataset']['target_value'])
            n_train_samples = len(x)
            mean_train *= n_train_samples
            std_train *= n_train_samples
            x_train.append(x)
            y_train.append(y)
            if not os.path.isfile(split_test_file):
                self.log(f"File not found:\n{split_test_file}"
                             f"\nNo test set is provided!", LogLevel.DEBUG)
                mean_test, std_test = copy.deepcopy(mean_train), copy.deepcopy(std_train)
                x, y = [], []
            else:
                x, y, mean_test, std_test = read_file(split_test_file)
            n_test_samples = len(x)
            mean_test *= n_test_samples
            std_test *= n_test_samples

            x_test.append(x)
            y_test.append(y)
            local_stats.append([[n_train_samples, mean_train.tolist(), std_train.tolist()],
                                [n_test_samples, mean_test.tolist(), std_test.tolist()]])
        self.store('x_train', x_train)
        self.store('x_test', x_test)
        self.store('y_train', y_train)
        self.store('y_test', y_test)
        return local_stats


@app_state(name="GlobalStats", role=Role.COORDINATOR)
class GlobalStats(AppState):
    def register(self):
        self.register_transition('WriteResults', Role.COORDINATOR)

    def run(self):
        aggregated_stats = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        print(aggregated_stats)
        self.update(progress=0.4)
        global_stats = []
        if self.load('method') == "variance":
            for train_split, test_split in aggregated_stats:
                n_train_samples, train_mean, train_std = train_split
                n_test_samples, test_mean, test_std = test_split
                if n_train_samples != 0:
                    train_mean = np.array(train_mean) / n_train_samples
                    train_std = np.array(train_std) / n_train_samples
                else:
                    train_mean = np.array(train_mean) * 0
                    train_std = np.array(train_std) * 0
                if n_test_samples != 0:
                    test_mean = np.array(test_mean) / n_test_samples
                    test_std = np.array(test_std) / n_test_samples
                else:
                    test_mean = np.array(test_mean) * 0
                    test_std = np.array(test_std) * 0
                global_stats.append([train_mean, train_std, test_mean, test_std])
            self.broadcast_data(data=global_stats)
        else:
            self.log(f"{self.load('method')} was not implemented as a normalization method.",
                         LogLevel.ERROR)
            self.update(state=op_state.ERROR)
        self.update(progress=0.5)
        return 'WriteResults'


@app_state(name='WriteResults', role=Role.BOTH)
class WriteResults(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self) -> str:
        global_stats = self.await_data(n=1, unwrap=True, is_json=False)
        progress = 0.5
        step = 0.5 / len(global_stats)
        for i, split_stats in enumerate(global_stats):
            x_train, x_test = \
                self.local_normalization(self.load('x_train')[i], self.load('x_test')[i], split_stats)
            self.write_results(x_train, x_test, i)
            progress += step
            self.update(progress=progress)
        self.update(progress=1.0)
        return 'terminal'

    def local_normalization(self, x_train, x_test, global_stats):
        if self.load('method') == "variance":
            normalized_x_train = np.subtract(x_train, global_stats[0]) / global_stats[1]
            if np.size(x_test) != 0:
                normalized_x_test = np.subtract(x_test, global_stats[2]) / global_stats[3]
            else:
                normalized_x_test = np.array([])
            return normalized_x_train.tolist(), normalized_x_test.tolist()

        self.log(f"{self.conmethod} was not implemented as a normalization method.", LogLevel.ERROR)
        self.update(state=op_state.ACTION)

    def write_results(self, x_train, x_test, i):
        save_numpy(file_name=self.load('output_files')['train'][i],
                   features=x_train,
                   labels=self.load('y_train')[i],
                   target=self.load('target_value'),
                   )
        if np.size(x_test) != 0:
            save_numpy(file_name=self.load('output_files')['test'][i],
                       features=x_test,
                       labels=self.load('y_test')[i],
                       target=self.load('target_value')
                       )
