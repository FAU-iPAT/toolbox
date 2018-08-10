from unittest import TestCase
from dieFFT.toolbox.callbacks import RunStatsCallback
from unittest.mock import MagicMock


class TestRunStatsCallback(TestCase):

    def setUp(self):
        self._rsc = RunStatsCallback()
        self._rsc.params = {'epochs': 5}

    def tearDown(self):
        self._rsc = None

    def test___init__(self):
        self.assertEqual(self._rsc._epoch_time_start, 0.0)
        self.assertEqual(self._rsc._runstats, {})

    def test_on_train_begin(self):
        self._rsc.on_train_begin({})
        rs = self._rsc.runstats
        self.assertEqual(set(rs.keys()), set(['time', 'lr']))
        self.assertEqual(rs['time'], [])
        self.assertEqual(rs['lr'], [])

    def test_on_epoch_begin(self):
        self._rsc.on_epoch_begin(3, {})
        self.assertNotEqual(self._rsc._epoch_time_start, 0.0)

    def test_on_epoch_end(self):
        self._rsc.on_train_begin({})
        # Test with no logs
        self._rsc.on_epoch_end(1, {})
        self.assertEqual(set(self._rsc.runstats.keys()), set(['time', 'lr']))
        self.assertEqual(len(self._rsc.runstats['time']), 1)
        self.assertEqual(len(self._rsc.runstats['lr']), 0)
        # Mock learning rate
        self._rsc.model = MagicMock()
        self._rsc.model.optimizer = MagicMock()
        self._rsc.model.optimizer.lr = MagicMock()
        self._rsc.model.optimizer.lr.eval = MagicMock(return_value=0.5)
        self._rsc.on_epoch_end(2, {})
        self.assertEqual(len(self._rsc.runstats['lr']), 1)
        self.assertEqual(self._rsc.runstats['lr'][0], 0.5)
        # Test additional log values
        self._rsc.on_epoch_end(3, {'one': 1.0, 'two': 2.0})
        self.assertEqual(set(self._rsc.runstats.keys()), set(['time', 'lr', 'one', 'two']))
        self.assertEqual(len(self._rsc.runstats['one']), 1)
        self.assertEqual(self._rsc.runstats['one'][0], 1.0)
        self.assertEqual(len(self._rsc.runstats['two']), 1)
        self.assertEqual(self._rsc.runstats['two'][0], 2.0)

    def test_runstats(self):
        x = 'abc'
        self._rsc._runstats = x
        self.assertEqual(self._rsc.runstats, x)
        self._rsc._runstats = {}

    def test_progress(self):
        prog = self._rsc.progress
        # todo: Check Keras/Tensorflow variable for viability?
        # self.fail('Check Keras/Tensorflow variable for viability?')
