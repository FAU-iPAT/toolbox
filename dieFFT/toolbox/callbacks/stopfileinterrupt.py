#! /opt/conda/bin/python3
""" File containing keras interrupt handler class for stopping on ".stop" file """

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


import os
from typing import Any
from keras.callbacks import Callback  # type: ignore


class StopFileInterrupt(Callback):
    """
    Class stopping keras training on found ".stop" file
    """

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        """
        Callback method to check for stopping after each epoch

        :param epoch: Current processed epoch
        :param logs: Logging data
        """
        if os.path.exists("./.stop"):
            os.remove("./.stop")
            print("")
            print("Interrupting training after epoch {0}!".format(epoch))
            self.model.stop_training = True
