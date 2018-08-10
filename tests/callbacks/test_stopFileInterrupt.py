import os
from unittest import TestCase
from unittest.mock import MagicMock
import contextlib
from io import StringIO
from dieFFT.toolbox.callbacks import StopFileInterrupt


class TestStopFileInterrupt(TestCase):

    def setUp(self):
        test_path = os.path.dirname(os.path.dirname(__file__))
        test_path = os.path.join(test_path, '_test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        self._cwd = os.getcwd()
        os.chdir(test_path)
        self._untouch()

    def tearDown(self):
        os.chdir(self._cwd)
        self._untouch()

    def _touch(self):
        open('.stop', 'a').close()

    def _untouch(self):
        if os.path.exists('.stop'):
            os.remove('.stop')

    def test_on_epoch_end(self):
        sfi = StopFileInterrupt()
        sfi.model = MagicMock()
        sfi.model.stop_training = False
        # Test for no stopping
        sfi.on_epoch_end(epoch=128, logs={})
        self.assertEqual(sfi.model.stop_training, False)
        # Test for stopping
        self._touch()
        io = StringIO()
        with contextlib.redirect_stdout(io):
            sfi.on_epoch_end(epoch=128, logs={})
        self.assertEqual(sfi.model.stop_training, True)
        self.assertEqual(os.path.exists('.stop'), False)

