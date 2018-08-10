from unittest import TestCase
from keras.layers import Input, Dense
from keras.models import Model as KerasModel
from dieFFT.toolbox.models import Model, model_from_config, model_from_json
import numpy as np


class TestModel(TestCase):

    def test_get_activations(self):
        i = Input(shape=(10,3))
        d = Dense(5, name='dense1')(i)
        d = Dense(7, name='dense2')(d)
        m = Model(i, d)
        a = m.get_activations(np.ones((5,10,3)), 'dense1')
        self.assertEqual(tuple(a[0].shape), (5,10,5))

    def test_get_sorted_weights(self):
        i = Input(shape=(10,3))
        d = Dense(5)(i)
        m = Model(i, d)
        w = m.get_sorted_weights(include_bias=False)
        self.assertEqual(len(w), 15)
        w = m.get_sorted_weights(include_bias=True)
        self.assertEqual(len(w), 20)

    def test_model_from_config(self):
        i = Input(shape=(10,3))
        d = Dense(5)(i)
        m = KerasModel(i, d)
        c = {'class_name': 'Model', 'config': m.get_config()}
        m2 = model_from_config(c)
        self.assertEqual(type(m2), Model)
        m3 = model_from_config(c, {'Model': KerasModel})
        self.assertEqual(type(m3), KerasModel)

    def test_model_from_json(self):
        i = Input(shape=(10,3))
        d = Dense(5)(i)
        m = KerasModel(i, d)
        j = m.to_json()
        m2 = model_from_json(j)
        self.assertEqual(type(m2), Model)
        m3 = model_from_json(j, {'Model': KerasModel})
        self.assertEqual(type(m3), KerasModel)
