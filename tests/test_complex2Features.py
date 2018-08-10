from unittest import TestCase
from dieFFT.toolbox import Complex2Features, c2f
import numpy as np
import numpy.testing as npt


class TestComplex2Features(TestCase):

    def test__data_to_dataset(self):
        # Test invalid data: empty tuple
        with self.assertRaises(ValueError):
            Complex2Features._data_to_dataset(tuple())
        # Test invalid data: shapeless type
        with self.assertRaises(AttributeError):
            Complex2Features._data_to_dataset(None)
        # Test invalid data: different shapes
        d1 = np.ones((10,256))
        d2 = np.zeros((128,64))
        d3 = np.zeros((10,256))
        with self.assertRaises(ValueError):
            Complex2Features._data_to_dataset((d1, d2))
        # Test valid data: single entry
        r = Complex2Features._data_to_dataset(d1)
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 1)
        # Test valid data: multiple entries
        r = Complex2Features._data_to_dataset((d1, d3))
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)

    def test__normalize_dataset(self):
        d_ones = np.ones((10,256))
        d_arange = np.reshape(np.arange(2560).astype(float), (10,256))
        d_max = np.reshape(np.arange(255, 10*256, 256), (10,1))
        dn_arange = d_arange / d_max
        dn_ones = d_ones / d_max
        # Test no normalization
        d = (d_ones, d_arange)
        r = Complex2Features._normalize_dataset((d), normalize=False)
        npt.assert_array_equal(d, r)
        # Test global normalization
        r = Complex2Features._normalize_dataset((d_ones, ), normalize=True)
        npt.assert_array_almost_equal(d_ones, r[0])
        r = Complex2Features._normalize_dataset((d_arange, ), normalize=True)
        npt.assert_array_almost_equal(dn_arange, r[0])
        r = Complex2Features._normalize_dataset((d_ones, d_arange), normalize=True)
        npt.assert_array_almost_equal(dn_ones, r[0])
        npt.assert_array_almost_equal(dn_arange, r[1])
        # Test one entry normalization
        r = Complex2Features._normalize_dataset((d_ones, d_arange), normalize=0)
        npt.assert_array_almost_equal(d_ones, r[0])
        npt.assert_array_almost_equal(d_arange, r[1])
        r = Complex2Features._normalize_dataset((d_ones, d_arange), normalize=1)
        npt.assert_array_almost_equal(dn_ones, r[0])
        npt.assert_array_almost_equal(dn_arange, r[1])

    def test__convert_data_entry(self):
        # Test valid result shape
        d = np.ones((10,256))
        r = Complex2Features._convert_data_entry(d, True, False, True, False)
        self.assertEqual(r.shape, (10,256,2))
        d = np.ones((17,128,9))
        r = Complex2Features._convert_data_entry(d, False, True, True, True)
        self.assertEqual(r.shape, (17,128,9,3))
        # Initialize test values
        d = np.arange(10)
        d = np.sin(d) + 1j * np.cos(d)
        d_re = np.reshape(np.real(d), d.shape + (1,))
        d_im = np.reshape(np.imag(d), d.shape + (1,))
        d_abs = np.reshape(np.abs(d), d.shape + (1,))
        d_phi = np.reshape(np.angle(d), d.shape + (1,))
        # Test single valued results
        r = Complex2Features._convert_data_entry(d, True, False, False, False)
        npt.assert_array_almost_equal(r, d_re)
        r = Complex2Features._convert_data_entry(d, False, True, False, False)
        npt.assert_array_almost_equal(r, d_im)
        r = Complex2Features._convert_data_entry(d, False, False, True, False)
        npt.assert_array_almost_equal(r, d_abs)
        r = Complex2Features._convert_data_entry(d, False, False, False, True)
        npt.assert_array_almost_equal(r, d_phi)
        # Test multiple valued result
        r = Complex2Features._convert_data_entry(d, True, True, True, False)
        npt.assert_array_almost_equal(r, np.concatenate((d_re, d_im, d_abs), axis=-1))

    def test__convert_dataset(self):
        # Initialize test values
        d = np.arange(10)
        d = np.sin(d) + 1j * np.cos(d)
        d_re = np.reshape(np.real(d), d.shape + (1,))
        # Test multiple data sets
        r = Complex2Features._convert_dataset((d, d), True, False, False, False)
        npt.assert_array_almost_equal(r, np.concatenate((d_re, d_re), axis=-1))

    def test_apply(self):
        for grouped in (True, False):
            # Test for valid shapes
            d = np.ones((10,256))
            r = Complex2Features.apply(d, False, True, True, False, False, grouped)
            self.assertEqual(r.shape, (10,256,2))
            r = Complex2Features.apply((d, -d), True, True, True, False, True, grouped)
            self.assertEqual(r.shape, (10,256,6))
            r = Complex2Features.apply((d, d, -2*d), 2, True, False, False, False, grouped)
            self.assertEqual(r.shape, (10,256,3))
            # Test values
            d = np.reshape(np.arange(2560), (10,256))
            d = np.sin(d) + 1j * np.cos(d)
            d_re = np.reshape(np.real(d), d.shape)
            r = Complex2Features.apply((d, 2*d), False, True, False, False, False, grouped)
            self.assertEqual(r.shape, (10,256,2))
            npt.assert_array_almost_equal(d_re, r[...,0])
            npt.assert_array_almost_equal(2*d_re, r[...,1])
        # Test ordering
        d = np.reshape(np.arange(2560), (10,256))
        d = np.sin(d) + 1j * np.cos(d)
        d_re = np.reshape(np.real(d), d.shape)
        d_im = np.reshape(np.imag(d), d.shape)
        r = Complex2Features.apply((d, 2*d), False, True, True, False, False, False)
        self.assertEqual(r.shape, (10, 256, 4))
        npt.assert_array_almost_equal(d_re, r[..., 0])
        npt.assert_array_almost_equal(2 * d_re, r[..., 1])
        npt.assert_array_almost_equal(d_im, r[..., 2])
        npt.assert_array_almost_equal(2 * d_im, r[..., 3])
        r = Complex2Features.apply((d, 2*d), False, True, True, False, False, True)
        self.assertEqual(r.shape, (10, 256, 4))
        npt.assert_array_almost_equal(d_re, r[..., 0])
        npt.assert_array_almost_equal(2 * d_re, r[..., 2])
        npt.assert_array_almost_equal(d_im, r[..., 1])
        npt.assert_array_almost_equal(2 * d_im, r[..., 3])

    def test_c2f(self):
        d = np.reshape(np.random.rand(256), (16, 16))
        r1 = Complex2Features.apply(d, False, True, True, False, True)
        r2 = c2f(d, False, True, True, False, True)
        npt.assert_array_almost_equal(r1, r2)

    def test__assemble_dataset(self):
        d1 = np.ones((10, 20))
        d2 = np.zeros((10, 20))
        d3 = np.reshape(np.arange(200), (10,20))
        r = Complex2Features._assemble_dataset((d1, d2, d3))
        self.assertTupleEqual(r.shape, (10,20,3))
        self.assertEqual(r[5, 17, 0], 1)
        self.assertEqual(r[7, 3, 1], 0)
        self.assertEqual(r[0, 13, 2], 13)

    def test__block_convert_dataset(self):
        # Initialize test values
        d = np.arange(10)
        d = np.sin(d) + 1j * np.cos(d)
        d_re = np.reshape(np.real(d), d.shape + (1,))
        d_im = np.reshape(np.imag(d), d.shape + (1,))
        # Test multiple data sets
        r = Complex2Features._block_convert_dataset((d, 2*d), True, True, False, False)
        npt.assert_array_almost_equal(r, np.concatenate((d_re, 2*d_re, d_im, 2*d_im), axis=-1))
