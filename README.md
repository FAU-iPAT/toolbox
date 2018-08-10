<a name="top"></a>
# Deep Interpretation Enhanced FFT (dieFFT) - Toolbox

## Collection of useful custom tools for training of neural networks

<hr>

- [Motivation](#intro)
- [Keras Model class extenstion](#model)
- [Keras training callback classes](#callbacks)
- [Complex2Features function](#c2f)
- [DataGenerator class](#datagenerator)




<hr><a name="intro"></a><a href="#top" style="float:right;">top</a>

# Motivation
<hr>

This repository contains a collection of different code snippets, classes and tools that we have been using during our different experiments in training neural networks using Keras with TensorFlow backend. This ever growing collection of tools ranges from very small functions with only a few lines of code to very complex classes encapsulating more extensive behaviours. On the functional level the tools range from simple data preparation functions, to more extensive data handling classes up to custom neural network layers for the Keras framework.

Almost all of the tools provided in this repository are a constant work-in-progress, and they are not the complete set of helper tools used in our experimentations. The provided tools of this repository represent the subset of functions from our tools library which are needed as a requirement for other repositories that we published, or are functions we think might be useful for other users too. The repository will be updated regularly as we complete new useful tools, or publish new repositories with extended dependencies.





<hr><a name="model"></a><a href="#top" style="float:right;">top</a>

# Keras Model class extension
<hr>

For a number of reasons direct replacements/extensions of the Keras ``Model`` class and its according loader functions ``model_from_json`` and ``model_from_config`` have been implemented. They can be used as direct replacements of the original Keras methods as shown in the following code sample.

```python
    from dieFFT.toolbox.models import Model, model_from_json

    # Generate a new model from keras layers
    model = Model(input_layer, output_layer)
    json_string = model.to_json()

    # Load a model from JSON string
    model = model_from_json(json_string)
    model.summary()    
```

The model class was extended with two additional methods which help in the investigation of the neural networks internal state:
- ``get_activations`` allows to track the activation values of hidden layers for a given input
- ``get_sorted_weights`` returns a sorted list of all network weights (e.g. for statistical analysis)

The ``model_from_json`` and ``model_from_config`` methods are extended to automatically replace the default Keras ``Model`` with this custom, extended ``Model`` class during loading of a neural network. Since it extends the Keras ``Model`` class this should work for all cases. Additionally those loader methods can be easily extended to automatically include custom layers to be loaded without manually notifying the loader process about them every time.





<hr><a name="callbacks"></a><a href="#top" style="float:right;">top</a>

# Keras training callback classes
<hr>

The repository contains a few callback classes which can be passed as arguments to the training methods of the Keras framework. If passed as arguments those classes perform certain actions at specific times during the training process. The provided classes extend the set of callbacks already contained in the Keras framework with some functionality that proved to be useful during our experimentations.

### StopFileInterrupt callback class

The ``StopFileInterrupt`` callback is a very basic class that simply checks for the existance of a ``./.stop`` file, and if found ends the training at the end of the current epoch. Its useful to perform a controlled premature stop to the training phase for programs running in the background.

```python
    from dieFFT.toolbox.callbacks import StopFileInterrupt

    # Pass callbacks to the fitting method of Keras
    model.fit(features, targets, callbacks=[StopFileInterrupt(),])
```

### RunStatsCallback class

The different fitting method of the Keras framework (like ``fit`` or ``fit_generator``) all return a history object at the end of the training which contains loss and accuracy values for each epoch. The same properties are also collected by the ``RunStatsCallback`` callback class, with the difference that the values of the previous epochs can already be accessed during the training. In addition to  those values returned in the history object, the ``RunStatsCallback`` class also collects information about the required time per epoch and logs the current learning rate (as far as available). Finally the callback also provides a ``progress`` property, which is a tensorflow variable containing the current progress of the training (current epoch / maximum number of epochs). This value can be useful for more advanced learning schedules.

```python
    from dieFFT.toolbox.callbacks import RunStatsCallback

    # Pass callbacks to the fitting method of Keras
    rsc = RunStatsCallback()
    history = model.fit(features, targets, callbacks=[rsc,])
    runstats = rsc.runstats
```





<hr><a name="c2f"></a><a href="#top" style="float:right;">top</a>

# Complex2Features function
<hr>

The ``c2f`` function and the encapsulating ``Complex2Features`` class provide a means to preprocess the input data for fourier spectral analysis problems. The fourier transformation of a signal calculates a complex valued result. But since the Keras/Tensorflow stack can't handle complex numbers naturally, the resulting data need to be processed into real valued data. Thus this ``c2f`` function transforms a set of complex valued data into the corresponding sets of real, imaginary and absolute components of the input data.

```python
    from dieFFT.toolbox import c2f
    
    features = c2f([complex_fourier])
```

If the input ``complex_fourier`` has a shape of ``(1024,)``, the resulting ``features`` would have a shape of ``(1024,3)``. If multiple inputs are passed as a tuple, their real, imaginary and absolute values are stacked. Thus two inputs of shape ``(1024,)`` result in a feature output of shape ``(1024,6)``. Using the default option, the resulting features for ``n`` inputs are stacked as follows:

- Real component input 1
- ...
- Real component input n
- Imaginary component input 1
- ...
- Imaginary component input n
- Absolute value input 1
- ...
- Absolute value input n

Setting the optional ``group_by_signal`` parameter to ``True`` will result in an ordering like this:

- Real component input 1
- Imaginary component input 1
- Absolute value input 1
- ...
- Real component input n
- Imaginary component input n
- Absolute value input n

Additionally by using the appropriate parameters, the feature output of the method can be further adopted (e.g. to include the phase, or exclude real, imaginary or absolute value, or to normalize the output). Have a look at the [function documentation](./dieFFT/toolbox/complex2features.py) comment to find the full list of available parameters.





<hr><a name="datagenerator"></a><a href="#top" style="float:right;">top</a>

# DataGenerator class
<hr>

Simply put, the ``DataGenerator`` class is a wrapper class for a generator which provides data for the ``*_generator`` style methods of the Keras framework. It is able to load data split across multiple files and provide them as continuous generator stream. It is also able to load data from multiple directories simultaneously, e.g. input data, target data and additional configuration data. The data is finally yielded as according tuples to feed into the Keras ``*_generator`` functions. During its handling of the data, a number of different operations can be performed. These include the following:
- Split the data into training, validation and testing sets; methods can be applied to arbitrary combinations of those parts
- Split the loaded data into batches of given size
- Shuffle the returned batches
- Count the number of batches across all input files
- Callback to preprocess the data loaded from files before further processing
- Using callback selection function to exclude certain data entries from being returned
- Further callback to preprocess the remaining data after selection
- Count the number of remaining sample after selection (for categorical also count samples per category)

A simple example of how to apply the ``DataGenerator`` class is provided below. For more extensive examples have a look at the training scripts in other of our repositories (e.g. [Tucana Training](https://github.com/FAU-iPAT/tucana/blob/master/training/train_tucana_v5.4.py)) where the ``DataGenerator`` class is applied.

```python
from dieFFT.toolbox import c2f, DataGenerator, DataGeneratorSelection, DataGeneratorEnhancement

# Define a preparation class to transform complex fourier transformation via c2f function into features vector
# (data enhancement will be applied before selection)
class DataPrepare(DataGeneratorEnhancement):

    def __init__(self):
        super(DataPrepare, self).__init__()
        
    def enhance(self, data_in, answer_in, config):
        data = c2f(data_in, normalize=0, real=True, imaginary=True, absolute=True)
        return [data], answer_in, config

# Define data selection class (In this example no actual selection is done)
class DataSelect(DataGeneratorSelection):

    def __init__(self):
        super(DataSelect, self).__init__()

    def select(self, idx, data, answer, config):
        valid = np.equal(idx, idx)
        return valid
        
# Define a data augmentation class (No further enhancement is performed in this example)
# (data enhancement will be applied after selection)
class DataEnhance(DataGeneratorEnhancement):

    def __init__(self):
        super(DataEnhance, self).__init__()

    def enhance(self, data, answer, config):
        return data, answer, config
        
# Initialize the generator class and pass all basic setup parameters
dg = DataGenerator(
    path_data=['./features/', './additional_features/'],
    path_answer='./targets/',
    path_config='./data_configuration/',
    file_format='batches_{:05d}.npy',
    file_limit=None,  # Automatically count all available files matching the file_format
    validation=0.2,
    testing=0.1,
    data_loader='numpy_dict',
    file_size=None,  # Get automatically from first data file
    batch_size=128,
)

# Create the actual generator object to be passed to the Keras *_generator methods
gen = dg.generator(
    dataset='training',  # Return the training part of the data split
    batch_size=None,  # Use previously defined DataGenerator batch size
    shuffle=True,
    prepend_idx=False,
    append_config=False,
    preparation=(DataPrepare()),
    selection=(DataSelect()),
    enhancement=(DataEnhance()),
)
gen_count = dg.batches(dataset='training', batch_size=None)

# Use the generator in training process
model.fit_generator(
    gen,
    steps_per_epoch=gen_count,
    max_q_size=15,
)
```

As mentioned before the ``DataGenerator`` class provides a wide variety of options how to preprocess data. Using the callback classes interface shown above, this class is quite generic and can be used for a wide range of applications with very different preprocessing pipelines. All the available methods of the classes and their parameters with according documentation can be found in the [class file](./dieFFT/toolbox/datagenerator.py).
