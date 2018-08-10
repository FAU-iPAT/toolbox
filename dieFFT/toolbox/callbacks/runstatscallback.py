#! /opt/conda/bin/python3
""" File containing keras callback class to collect runstats of the training process """

# Copyright 2018 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from typing import Any, Dict, List
from keras.callbacks import Callback  # type: ignore
import keras.backend as backend  # type: ignore


_TRunStats = Dict[str, List[float]]  # pylint: disable=invalid-name


class RunStatsCallback(Callback):
    """
    Callback class to log runstats of the keras model

    This class stores the default history of the optimizer plus some
    additional values. Those include value for each epoch for:
    - Time required
    - Base learning rate of the optimizer
    At the end of the training those values call be accessed via the
    runstats property.
    """

    def __init__(self) -> None:
        """
        Class initialization
        """
        super(RunStatsCallback, self).__init__()
        self._runstats = {}  # type: _TRunStats
        self._epoch_time_start = 0.0
        # Allocate epoch and progress info tensors
        self._epoch = backend.variable(0.0, dtype='float32', name='RunStatsCallbackEpoch')
        self._max_epochs = backend.variable(1.0, dtype='float32', name='RunStatsCallbackMaxEpochs')
        self._progress = self._epoch / self._max_epochs

    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        """
        Callback method to setup at beginning of training

        :param logs: Log data from keras
        """
        self._runstats = {'time': [], 'lr': []}
        epochs = self.params['epochs'] if self.params['epochs'] else 1.0
        backend.set_value(self._max_epochs, epochs)
        backend.set_value(self._epoch, 0.0)

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """
        Callback method called at beginning of each epoch

        :param epoch: Epoch to be started
        :param logs: Log data from keras
        """
        self._epoch_time_start = time.time()
        backend.set_value(self._epoch, epoch)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """
        Callback method called at the end of each epoch

        :param epoch: Epoch to be ended
        :param logs: Log data from keras
        """
        backend.set_value(self._epoch, epoch+1)
        # Store default history data
        if logs:
            for name in logs:
                if name not in self._runstats:
                    self._runstats[name] = []
                self._runstats[name].append(logs[name])
        # Additionally store time required
        self._runstats['time'].append(time.time() - self._epoch_time_start)
        # Additionally store base learning rate of the optimizer
        try:
            learning_rate = self.model.optimizer.lr
            self._runstats['lr'].append(backend.get_value(learning_rate))
        except AttributeError:
            pass

    @property
    def runstats(self) -> _TRunStats:
        """
        runstats property

        :return: runstats dictionary
        """
        return self._runstats

    @property
    def progress(self):
        """
        Progress tensor property

        :return: progress tensor
        """
        return self._progress
