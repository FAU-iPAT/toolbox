import os
from unittest import TestCase
from dieFFT.toolbox import DataGenerator, DataGeneratorEnhancement, DataGeneratorSelection
import numpy as np
import numpy.testing as npt
import contextlib
from io import StringIO
import json
import gzip


class TestDataGenerator(TestCase):

    @classmethod
    def setUpClass(cls):
        test_path = os.path.dirname(os.path.dirname(__file__))
        test_path = os.path.join(test_path, '_test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        cls._cwd = os.getcwd()
        os.chdir(test_path)
        filename = 'batch_{0:05d}.npy'.format(0)
        data = np.ones((16,32))
        data[:,0] = np.arange(16)+1
        np.save(filename, data)
        filename2 = 'batch_{0:05d}.npy'.format(1)
        if os.path.exists(filename2):
            os.remove(filename2)

    @classmethod
    def tearDownClass(cls):
        filename = 'batch_{0:05d}.npy'.format(0)
        if os.path.exists(filename):
            os.remove(filename)
        os.chdir(cls._cwd)

    def setUp(self):
        self.dg = DataGenerator(path_data=['./'])

    def tearDown(self):
        del self.dg

    def test__print(self):
        dg = self.dg
        dg._verbose = 2
        io = StringIO()
        with contextlib.redirect_stdout(io):
            dg._print('HELP!', 1)
            dg._print('ME!', 3)
        self.assertEqual(io.getvalue(), 'HELP!'+"\n")

    def test__validate_path(self):
        dg = self.dg
        result = dg._validate_path(None)
        self.assertEqual(result, [])
        result = dg._validate_path('./')
        self.assertEqual(result, ['./'])
        result = dg._validate_path(['./', './'])
        self.assertEqual(result, ['./', './'])
        with self.assertRaises(FileNotFoundError):
            dg._validate_path('../')

    def test__validate_data_loader(self):
        dg = self.dg
        with self.assertRaises(ValueError):
            dg._validate_data_loader(3)
        def func(filename):
            return 37, [7,8,6]
        dl = dg._validate_data_loader(func)
        self.assertEqual(dl, func)
        self.assertEqual(dg._validate_data_loader('numpy'), dg._data_loader_numpy)
        self.assertEqual(dg._validate_data_loader('numpy_dict'), dg._data_loader_numpy_dict)
        self.assertEqual(dg._validate_data_loader('json'), dg._data_loader_json)
        self.assertEqual(dg._validate_data_loader('json+gzip'), dg._data_loader_json_gzip)
        with self.assertRaises(ValueError):
            dg._validate_data_loader('ERROR_NAME')

    def test__data_loader_numpy(self):
        dg = self.dg
        count, data = dg._data_loader_numpy('./batch_{0:05d}.npy'.format(0))
        self.assertEqual(count, 16)
        self.assertEqual(data.shape, (16, 32))

    def test__data_loader_numpy_dict(self):
        dg = self.dg
        count, data = dg._data_loader_numpy_dict('./batch_{0:05d}.npy'.format(0))
        self.assertEqual(count, 16)
        self.assertEqual(data.shape, (16, 32))
        data = {'demo': data}
        np.save('./numpy_dict.npy', data)
        count, data = dg._data_loader_numpy_dict('./numpy_dict.npy')
        self.assertEqual(count, 1)
        self.assertEqual(data['demo'].shape, (16, 32))
        os.remove('./numpy_dict.npy')

    def test__data_loader_json(self):
        dg = self.dg
        data = np.reshape(np.arange(128), (8,16)).tolist()
        with open('./data.json', 'w') as file:
            json.dump(data, file)
            file.close()
        count, data = dg._data_loader_json('./data.json')
        self.assertEqual(count, 8)
        self.assertEqual(list(data[1]), [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
        os.remove('./data.json')

    def test__data_loader_json_gzip(self):
        dg = self.dg
        data = np.reshape(np.arange(128), (8,16)).tolist()
        with gzip.open('./data.json.gzip', 'w') as file:
            value = json.dumps(data)
            file.write(value.encode())
            file.close()
        count, data = dg._data_loader_json_gzip('./data.json.gzip')
        self.assertEqual(count, 8)
        self.assertEqual(list(data[1]), [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
        os.remove('./data.json.gzip')

    def test__get_valid_testing_path(self):
        dg = self.dg
        self.assertEqual(dg._get_valid_testing_path(), './')
        dg._data = []
        dg._answer = ['./']
        self.assertEqual(dg._get_valid_testing_path(), './')
        dg._answer = []
        dg._config = ['./']
        self.assertEqual(dg._get_valid_testing_path(), './')
        dg._config = []
        with self.assertRaises(ValueError):
            dg._get_valid_testing_path()

    def test__count_files(self):
        dg = self.dg
        self.assertEqual(dg._count_files(), 1)
        dg._data = ['../']
        self.assertEqual(dg._count_files(), 0)

    def test__get_file_size(self):
        dg = self.dg
        self.assertEqual(dg._get_file_size(), 16)

    def test__split_count(self):
        dg = self.dg
        dg._file_count, dg._validation, dg._testing = 10, 0.2, 0.3
        self.assertTupleEqual(dg._split_count(), (5, 2, 3))
        dg._file_count, dg._validation, dg._testing = 10, 0.0, 0.0
        self.assertTupleEqual(dg._split_count(), (10, 0, 0))
        dg._file_count, dg._validation, dg._testing = 10, 0.0, 0.3
        self.assertTupleEqual(dg._split_count(), (7, 0, 3))
        dg._file_count, dg._validation, dg._testing = 10, 0.4, 0.0
        self.assertTupleEqual(dg._split_count(), (6, 4, 0))

    def test__batches_per_file(self):
        dg = self.dg
        dg._file_size = 16
        self.assertEqual(dg._batches_per_file(1), 16)
        self.assertEqual(dg._batches_per_file(2), 8)
        self.assertEqual(dg._batches_per_file(8), 2)
        self.assertEqual(dg._batches_per_file(128), 1)
        self.assertEqual(dg._batches_per_file(7), 3)

    def test__print_summary(self):
        dg = self.dg
        dg._answer = ['./']
        dg._config = [',/']
        io = StringIO()
        with contextlib.redirect_stdout(io):
            dg._print_summary(5)

    def test__get_file_idxs(self):
        dg = self.dg
        dg._file_count, dg._validation, dg._testing = 10, 0.2, 0.3
        self.assertEqual(dg._get_file_idxs('training'), [0,1,2,3,4])
        self.assertEqual(dg._get_file_idxs('validation'), [5,6])
        self.assertEqual(dg._get_file_idxs('testing'), [7,8,9])
        self.assertEqual(dg._get_file_idxs('all'), [0,1,2,3,4,5,6,7,8,9])
        self.assertEqual(dg._get_file_idxs('training+testing'), [0,1,2,3,4,7,8,9])
        self.assertEqual(dg._get_file_idxs('validation+testing'), [5,6,7,8,9])

    def test__shuffle_indices(self):
        dg = self.dg
        idx = np.arange(128)
        result = dg._shuffle_indices(idx, False)
        npt.assert_allclose(idx, result)
        count = 0
        for _ in range(1024):
            result = dg._shuffle_indices(list(idx), True)
            count += np.allclose(result, idx)
        self.assertLess(count, 3)

    def test__load_files(self):
        dg = self.dg
        idx, data, answer, config = dg._load_files(0, True)
        self.assertEqual(list(idx), list(np.arange(16)))
        self.assertEqual(list(data[0][:,0]), list(np.arange(16)+1))
        self.assertEqual(answer, [])
        self.assertEqual(config, [])
        idx, data, answer, config = dg._load_files(0, False)
        self.assertEqual(config, [])
        dg._answer = ['./']
        dg._config = ['./']
        idx, data, answer, config = dg._load_files(0, True)
        self.assertEqual(list(data[0][:,0]), list(np.arange(16)+1))
        self.assertEqual(list(answer[0][:,0]), list(np.arange(16)+1))
        self.assertEqual(list(config[0][:,0]), list(np.arange(16)+1))

    def test__validate_enhancements(self):
        dg = self.dg
        self.assertIsInstance(dg._validate_enhancements('abc'), list)
        self.assertIsInstance(dg._validate_enhancements(['abc']), list)
        result = dg._validate_enhancements(['abc', 'def'])
        self.assertEqual(result[0], 'abc')
        self.assertEqual(result[1], 'def')
        result = dg._validate_enhancements(('abc', 'def'))
        self.assertEqual(result[0], 'abc')
        self.assertEqual(result[1], 'def')
        result = dg._validate_enhancements('ijk')
        self.assertEqual(result[0], 'ijk')
        enh = DataGeneratorEnhancement()
        self.assertIsInstance(dg._validate_enhancements(enh), list)

    def test__validate_selections(self):
        dg = self.dg
        self.assertIsInstance(dg._validate_selections('abc'), list)
        self.assertIsInstance(dg._validate_selections(['abc']), list)
        result = dg._validate_selections(['abc', 'def'])
        self.assertEqual(result[0], 'abc')
        self.assertEqual(result[1], 'def')
        result = dg._validate_selections(('abc', 'def'))
        self.assertEqual(result[0], 'abc')
        self.assertEqual(result[1], 'def')
        result = dg._validate_selections('ijk')
        self.assertEqual(result[0], 'ijk')
        sel = DataGeneratorSelection()
        self.assertIsInstance(dg._validate_selections(sel), list)

    def test__apply_enhancements(self):
        dg = self.dg
        def enh(d, a, c):
            return d+a, d-a, c
        class enhc(DataGeneratorEnhancement):
            def enhance(self, d, a, c):
                return d + a, d - a, c
        data = np.ones((10,))
        answer = np.arange(10)
        config = np.zeros((10,))
        d, a, c = dg._apply_enhancements([enh], data, answer, config)
        self.assertEqual(list(d), [1,2,3,4,5,6,7,8,9,10])
        self.assertEqual(list(a), [1,0,-1,-2,-3,-4,-5,-6,-7,-8])
        self.assertEqual(list(c), [0,0,0,0,0,0,0,0,0,0])
        d, a, c = dg._apply_enhancements([enh, enh], data, answer, config)
        self.assertEqual(list(d), [2,2,2,2,2,2,2,2,2,2])
        self.assertEqual(list(a), [0,2,4,6,8,10,12,14,16,18])
        self.assertEqual(list(c), [0,0,0,0,0,0,0,0,0,0])
        data = np.ones((10,))
        answer = np.arange(10)
        config = np.zeros((10,))
        d, a, c = dg._apply_enhancements([enhc()], data, answer, config)
        self.assertEqual(list(d), [1,2,3,4,5,6,7,8,9,10])
        self.assertEqual(list(a), [1,0,-1,-2,-3,-4,-5,-6,-7,-8])
        self.assertEqual(list(c), [0,0,0,0,0,0,0,0,0,0])

    def test__apply_selection_vector(self):
        dg = self.dg
        data = np.arange(8)
        vector = data > 3
        r = dg._apply_selection_vector(data, vector)
        self.assertEqual(list(r), [4,5,6,7])
        r = dg._apply_selection_vector(list(data), vector)
        self.assertEqual(list(r), [4,5,6,7])
        data = {'data': np.arange(8), 'zeros': np.zeros((8,)), 'data2': -np.arange(8)}
        r = dg._apply_selection_vector(data, vector)
        self.assertIsInstance(r, dict)
        self.assertSetEqual(set(r.keys()), set(['data', 'data2', 'zeros']))
        self.assertEqual(list(r['data']), [4,5,6,7])
        self.assertEqual(list(r['zeros']), [0,0,0,0])
        self.assertEqual(list(r['data2']), [-4,-5,-6,-7])
        data = {'data': {'data': np.arange(8)}, 'zeros': np.zeros((8,))}
        r = dg._apply_selection_vector(data, vector)
        self.assertIsInstance(r, dict)
        self.assertSetEqual(set(r.keys()), set(['data', 'zeros']))
        self.assertIsInstance(r['data'], dict)
        self.assertSetEqual(set(r['data'].keys()), set(['data']))
        self.assertEqual(list(r['data']['data']), [4,5,6,7])
        self.assertEqual(list(r['zeros']), [0,0,0,0])

    def test__apply_selections(self):
        dg = self.dg
        def filter(i, d, a, c):
            return i > 2
        def filter2(i, d, a, c):
            return i < 8
        class filterc(DataGeneratorSelection):
            def select(self, i, d, a, c):
                return i > 2
        idx = np.arange(10)
        data = [np.ones((10,))]
        answer = [np.arange(10)]
        config = [np.zeros((10,))]
        i, d, a, c = dg._apply_selections([filter], idx, data, answer, config)
        self.assertEqual(list(i), [3,4,5,6,7,8,9])
        idx = np.arange(10)
        data = [np.ones((10,))]
        answer = [np.arange(10)]
        config = [np.zeros((10,))]
        i, d, a, c = dg._apply_selections([filter, filter2], idx, data, answer, config)
        self.assertEqual(list(i), [3,4,5,6,7])
        self.assertEqual(list(d[0]), [1,1,1,1,1])
        self.assertEqual(list(a[0]), [3,4,5,6,7])
        self.assertEqual(list(c[0]), [0,0,0,0,0])
        idx = np.arange(10)
        data = [np.ones((10,))]
        answer = [np.arange(10)]
        config = [np.zeros((10,))]
        i, d, a, c = dg._apply_selections([filterc()], idx, data, answer, config)
        self.assertEqual(list(i), [3,4,5,6,7,8,9])

    def test__delistify(self):
        data = np.ones((12,15))
        result = self.dg._delistify(data)
        self.assertIsInstance(result, np.ndarray)
        data = [data, ]
        result = self.dg._delistify(data)
        self.assertIsInstance(result, np.ndarray)

    def test__apply_slicing(self):
        dg = self.dg
        data = np.arange(8)
        r = dg._apply_slicing(data, 2, 5)
        self.assertEqual(list(r), [2,3,4])
        data = {'data': np.arange(8)}
        r = dg._apply_slicing(data, 2, 5)
        self.assertIsInstance(r, dict)
        self.assertSetEqual(set(r.keys()), set(['data']))
        self.assertEqual(list(r['data']), [2,3,4])
        data = {'data': {'inner': np.arange(8)}}
        r = dg._apply_slicing(data, 2, 5)
        self.assertIsInstance(r, dict)
        self.assertSetEqual(set(r.keys()), set(['data']))
        self.assertIsInstance(r['data'], dict)
        self.assertSetEqual(set(r['data'].keys()), set(['inner']))
        self.assertEqual(list(r['data']['inner']), [2,3,4])

    def test_batches(self):
        dg = self.dg
        dg._file_count = 10
        dg._file_size = 128
        dg._validation = 0.2
        dg._testing = 0.3
        self.assertEqual(dg.batches('training', 128), 5)
        self.assertEqual(dg.batches('training', 64), 2*5)
        self.assertEqual(dg.batches('training', 32), 4*5)
        self.assertEqual(dg.batches('training+testing', 128), 8)

    def test_count(self):
        dg = self.dg
        self.assertEqual(dg.count('all'), 16)
        def filter(i,d,a,c):
            return i > 5
        self.assertEqual(dg.count('all', selection=[filter]), 10)
        c, cats = dg.count('all', categorical_count=True)
        self.assertEqual(c, 16)
        self.assertIsInstance(cats, np.ndarray)

    def test_generator(self):
        dg = self.dg
        dg._answer = ['./']
        dg._config = ['./']
        gen = dg.generator('all', batch_size=1, append_config=True, prepend_idx=True, shuffle=False)
        for idx in range(16):
            i, d, a, c = next(gen)
            self.assertEqual(d[0,0], idx+1)
            self.assertEqual(i[0], idx)
        i, d, a, c = next(gen)
        self.assertEqual(d[0,0], 1)
        self.assertEqual(i[0], 0)

    def test___init__(self):
        with self.assertRaises(ValueError):
            dg = DataGenerator(path_data=['./'], validation=0.6, testing=0.8)
        io = StringIO()
        with contextlib.redirect_stdout(io):
            dg = DataGenerator(path_data=['./'], verbose=3)

    def test__clear_cache(self):
        # self.dg._nocache = True
        # self.dg._clear_cache()
        # todo: Systemcall only applicable to linux, indended for Jetsons
        pass


class TestDataGeneratorSelection(TestCase):

    def test_select(self):
        sel = DataGeneratorSelection()
        with self.assertRaises(NotImplementedError):
            sel.select(None, None, None, None)
        with self.assertRaises(NotImplementedError):
            sel(None, None, None, None)


class TestDataGeneratorEnhancement(TestCase):

    def test_enhance(self):
        enh = DataGeneratorEnhancement()
        with self.assertRaises(NotImplementedError):
            enh.enhance(None, None, None)
        with self.assertRaises(NotImplementedError):
            enh(None, None, None)
