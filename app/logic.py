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
import os
import shutil
import threading
import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

jsonpickle_numpy.register_handlers()


class AppLogic:
    """ Implementing the workflow for FeatureCloud platform

    Attributes
    ----------
    status_available: bool
    status_finished: bool
    id:
    coordinator: bool
    clients:
    data_incoming: list
    data_outgoing: list
    thread:
    iteration: int
    progress: str
    INPUT_DIR: str
    OUTPUT_DIR: str
    models: set
    mode: str
    dir: str
    splits: dict
    test_splits: dict
    parameters: dict
    mean_std: dict
    iter_counter: int
    workflows_states: dict
    coordinator_class: ClientModels
    client_class: ClientModels
    init_params_from_server: dict
    max_iter_reached: bool
    states: dict

    Methods
    -------
    handle_setup(client_id, coordinator, clients)
    handle_incoming(data)
    handle_outgoing()
    app_flow()
    send_to_server(data_to_send)
    wait_for_server()
    broadcast(data)
    read_config(config_file)
    read_input(path)
    local_preprocess(model, features, labels, init_from_server)
    global_preprocess(model, features, labels, mean_std)
    local_computation(model, parameters)
    global_aggregation(model, parameters)
    write_results(model, parameters, output_path)
    lazy_initialization(mode, dir)
    finalize_config()

    """

    def __init__(self):

        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.mode = None
        self.dir = None
        self.splits = {}
        self.test_splits = {}
        self.global_stats = {}
        self.workflows_states = {}

        # === States ===
        self.states = {"state_initializing": 1,
                       "state_read_input": 2,
                       "state_wait_for_global_stats": 3,
                       "state_global_stats": 4,
                       "state_local_normalization": 5,
                       "state_writing_results": 6,
                       "state_finishing": 7
                       }

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # Initial state
        state = self.states["state_initializing"]
        self.progress = 'initializing...'

        while True:
            print(f"{bcolors.STATE}{list(self.states.keys())[list(self.states.values()).index(state)]}{bcolors.ENDC}")
            if state == self.states["state_initializing"]:
                if self.id is not None:  # Test if setup has happened already
                    state = self.states["state_read_input"]

            if state == self.states["state_read_input"]:
                self.progress = "Config..."
                self.read_config(self.INPUT_DIR + '/config.yml')
                self.finalize_config()
                data_to_send = {}
                for split in self.splits.keys():
                    print(f'{bcolors.SPLIT}Read {split}{bcolors.ENDC}')
                    x_train, y_train, x_test, y_test, local_stats = self.read_input(split)
                    data_to_send[split] = local_stats
                    self.splits[split] = [x_train, y_train]
                    self.test_splits[split] = [x_test, y_test]
                self.send_to_server(data_to_send)
                if self.coordinator:
                    state = self.states["state_global_stats"]
                else:
                    state = self.states["state_wait_for_global_stats"]

            if state == self.states["state_wait_for_global_stats"]:
                self.progress = 'wait for global stats'
                decoded_data = self.wait_for_server()
                if decoded_data is not None:
                    print(f"{bcolors.SEND_RECEIVE}Received stats from coordinator.{bcolors.ENDC}")
                    self.global_stats = decoded_data
                    state = self.states["state_local_normalization"]

            if state == self.states["state_global_stats"]:
                self.progress = 'average local stats...'
                if len(self.data_incoming) == len(self.clients):
                    for stats, split in self.get_clients_data():
                        self.global_stats[split] = self.global_aggregation(stats)
                    self.broadcast(self.global_stats)
                    state = self.states["state_local_normalization"]

            if state == self.states["state_local_normalization"]:
                self.progress = 'local normalization'
                for split in self.splits.keys():
                    print(f'Preprocess {split}')
                    self.splits[split][0], self.test_splits[split][0] = \
                        self.local_preprocess(self.splits[split][0],
                                              self.test_splits[split][0],
                                              self.global_stats[split]
                                              )

                state = self.states["state_writing_results"]

            if state == self.states["state_writing_results"]:
                self.progress = "write"
                for split in self.splits.keys():
                    path = split.replace("input", "output")
                    self.write_results(self.splits[split], self.test_splits[split], path)
                if self.coordinator:
                    self.data_incoming.append('DONE')
                    state = self.states["state_finishing"]
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True

                    break

            if state == self.states["state_finishing"]:
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            time.sleep(1)

    def send_to_server(self, data_to_send):
        """  Will be called only for clients
            to send their parameters or locally computed
             mean and standard deviation for the coordinator

        Parameters
        ----------
        data_to_send: list

        """
        data_to_send = jsonpickle.encode(data_to_send)
        if self.coordinator:
            self.data_incoming.append(data_to_send)
        else:
            self.data_outgoing = data_to_send
            self.status_available = True
            print(f'{bcolors.SEND_RECEIVE} [CLIENT] Sending data to coordinator. {bcolors.ENDC}', flush=True)

    def get_clients_data(self):
        """ Will be called only for the coordinator
            to get all the clients communicated data
            for each split, corresponding clients' data will be yield back.

        Returns
        -------
        clients_data: list
        split: str
        """
        print(f"{bcolors.SEND_RECEIVE} Received data of all clients. {bcolors.ENDC}")
        data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
        self.data_incoming = []
        for split in self.splits.keys():
            print(f'{bcolors.SPLIT} Get {split} {bcolors.ENDC}')
            clients_data = []
            for client in data:
                clients_data.append(client[split])
            yield clients_data, split

    def wait_for_server(self):
        """ Will be called only for clients
            to wait for server to get
            some globally shared data.

        Returns
        -------
        None or list
            in case no data received None will be returned
            to signal the state!
        """
        if len(self.data_incoming) > 0:
            data_decoded = jsonpickle.decode(self.data_incoming[0])
            self.data_incoming = []
            return data_decoded
        return None

    def broadcast(self, data):
        """ will be called only for the coordinator after
            providing data that should be broadcast to clients

        Parameters
        ----------
        data: list

        """
        data_to_broadcast = jsonpickle.encode(data)
        self.data_outgoing = data_to_broadcast
        self.status_available = True
        print(f'{bcolors.SEND_RECEIVE} [COORDINATOR] Broadcasting data to clients. {bcolors.ENDC}', flush=True)

    def read_config(self, config_file):
        """ should be overridden!
            reads the config file.
            calls the lazy_initialization method!

        Parameters
        ----------
        config_file: string
            path to the config.yaml file!

        Raises
        ------
        NotImplementedError
        """
        NotImplementedError("read_config method in Applogic class is not implemented!")

    def read_input(self, path):
        """ should be overridden
        called for both clients and coordinator.
        to read train set
        coordinator also reads the test set
        for evaluation purpose after aggregation step!
        for reading input files, path to the file can made
        in this way:
        train_path = path + "/" + self.train_filename
        test_path = path + "/" + self.test_filename

        Parameters
        ----------
        path: str
            for one application it would be "/mnt/input"


        Returns
        -------
        x_train: numpy.array
        y_train: numpy.array
        x_test: numpy.array
        y_test: numpy.array
        local_stats: dict


        Raises
        ------
        NotImplementedError
        """
        NotImplementedError("preprocess method in Applogic class is not implemented!")

    def local_preprocess(self, x_train, x_test, global_stats):
        """ should be overridden!
            called for clients
            to initialized their models from globally shared wights.
            Also, to normalize their data based on the global statistics!

        Parameters
        ----------
        x_train: numpy.array
        x_test: numpy.array
        global_stats: dict

        Returns
        -------
        normalized_train_set: numpy.array
        normalized_test_set: numpy.array


        Raises
        ------
        NotImplementedError
        """

        NotImplementedError("preprocess method in Applogic class is not implemented!")

    def global_aggregation(self, stats):
        """ should be overridden!
            only called for the coordinator
            averaging clients mean and standard deviation
            and share the global mean and standard deviation
            with clients.

        Parameters
        ----------
        stats: dict

        Returns
        -------
        global_stats: dict


        Raises
        ------
        NotImplementedError
        """

        NotImplementedError("preprocess method in Applogic class is not implemented!")

    def write_results(self, train_set, test_set, output_path):
        """ should be overridden!
            writing results, e.g., predictions of the test set,
             into output directory.

        Parameters
        ----------
        train_set: list of numpy.array
        test_set: list of numpy.array
        output_path: str

        Raises
        ------
        NotImplementedError
        """
        NotImplementedError("write_results method in Applogic class is not implemented!")

    def lazy_initialization(self, mode, dir):
        """

        Parameters
        ----------
        mode: str
        dir: str
        """
        self.mode = mode
        self.dir = dir

    def finalize_config(self):
        """

        Returns
        -------

        """
        if self.mode == "directory":
            self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
            self.test_splits = dict.fromkeys(self.splits.keys())
            self.workflows_states = dict.fromkeys(self.splits.keys())
        else:
            self.splits[self.INPUT_DIR] = None
            self.test_splits[self.INPUT_DIR] = None
            self.workflows_states[self.INPUT_DIR] = None

        for split in self.splits.keys():
            os.makedirs(split.replace("/input", "/output"), exist_ok=True)
        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)


class TextColor:
    def __init__(self, color):
        if color:
            self.SEND_RECEIVE = '\033[95m'
            self.STATE = '\033[94m'
            self.SPLIT = '\033[96m'
            self.VALUE = '\033[92m'
            self.WARNING = '\033[93m'
            self.FAIL = '\033[91m'
            self.ENDC = '\033[0m'
            self.BOLD = '\033[1m'
            self.UNDERLINE = '\033[4m'
        else:
            self.SEND_RECEIVE = ''
            self.STATE = ''
            self.SPLIT = ''
            self.VALUE = ''
            self.WARNING = ''
            self.FAIL = ''
            self.ENDC = ''
            self.BOLD = ''
            self.UNDERLINE = ''


bcolors = TextColor(color=False)
