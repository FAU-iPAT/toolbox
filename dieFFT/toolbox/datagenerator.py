#! /opt/conda/bin/python3
""" Data generator class to provide data for *_generator keras methods """

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
import math
from typing import Union, List, Tuple, Generator, Callable, Any
import json
import gzip
import numpy as np


_TPathList = List[str]
_TPath = Union[str, _TPathList]

_TData = np.ndarray
_TDataList = List[_TData]

_TEnhancementFunctionReturn = Tuple[_TDataList, _TDataList, _TDataList]
_TEnhancementFunction = Callable[[_TDataList, _TDataList, _TDataList], _TEnhancementFunctionReturn]
_TEnhancementList = List[_TEnhancementFunction]
_TEnhancement = Union[_TEnhancementFunction, _TEnhancementList]

_TSelectionFunctionReturn = np.ndarray
_TSelectionFunction = Callable[[np.ndarray, _TDataList, _TDataList, _TDataList], _TSelectionFunctionReturn]
_TSelectionList = List[_TSelectionFunction]
_TSelection = Union[_TSelectionFunction, _TSelectionList]

_TDataLoaderFunction = Callable[[str], Tuple[int, _TData]]
_TDataLoader = Union[_TDataLoaderFunction, str]


class DataGeneratorSelection:  # pylint: disable=too-few-public-methods
    """
    Abstract base class for selection classes which can be used instead
    of a selection method
    """

    def select(  # pylint: disable=redundant-returns-doc
            self,
            idx: np.ndarray,
            data: _TDataList,
            answer: _TDataList,
            config: _TDataList
    ) -> np.ndarray:
        """
        Abstract method to override which handles the data selection

        :param idx: Indices of the data entries
        :param data: List of data to apply selection on
        :param answer: List of answers to apply selection on
        :param config: List of configs to apply selection on
        :return: Selection bool vector
        :raises NotImplementedError: Abstract method to override
        """
        raise NotImplementedError('Abstract method: implement in derived classes!')

    def __call__(self, *args, **kwargs):
        """
        Callable callback method dispatching call to enhance method

        :param args: Argument
        :param kwargs: Keyword arguments
        :return: Returned result
        """
        return self.select(*args, **kwargs)


class DataGeneratorEnhancement:  # pylint: disable=too-few-public-methods
    """
    Abstract base class for enhancement classes which can be used instead
    of an enhancement method
    """

    def enhance(  # pylint: disable=redundant-returns-doc
            self,
            data: _TDataList,
            answer: _TDataList,
            config: _TDataList
    ) -> Tuple[_TDataList, _TDataList, _TDataList]:
        """
        Abstract method to override which handles the data enhancements

        :param data: List of data to be enhanced
        :param answer: List of answers to be enhanced
        :param config: List of configs to be enhanced
        :return: Tuple of enhanced data, answers, configs
        :raises NotImplementedError: Abstract method to override
        """
        raise NotImplementedError('Abstract method: implement in derived classes!')

    def __call__(self, *args, **kwargs):
        """
        Callable callback method dispatching call to enhance method

        :param args: Argument
        :param kwargs: Keyword arguments
        :return: Returned result
        """
        return self.enhance(*args, **kwargs)


class DataGenerator:  # pylint: disable=too-many-instance-attributes
    """
    Generator class for serving data to *_generator methods of keras
    """

    def _print(self, text: str, verbosity: int = 3) -> None:
        """
        Print some text on verbose level

        :param text: Text to be printed
        :param verbosity: Minimum verbosity to print
        """
        if self._verbose >= verbosity:
            print(text)

    def _validate_path(self, path: _TPath) -> _TPathList:
        """
        Transform input path/list of paths to validated path list

        :param path: Path str or list of paths to be checked
        :return: Validated list of paths
        :raises FileNotFoundError: Path not contained valid data
        """
        self._print('DataGenerator: _validate_path')
        result = []
        if path is not None:
            if isinstance(path, str):
                path = [path, ]
            for current in path:
                if os.path.isfile(current + self._format.format(0)) is False:
                    raise FileNotFoundError('Directory does not exist or does not contain data files ' + current)
                result.append(current)
        return result

    def _validate_data_loader(self, data_loader: _TDataLoader) -> _TDataLoaderFunction:
        """
        Validate data_loader and map to callable data loader function

        :param data_loader: String or function for the data loader
        :return: Validated data loader function
        :raises ValueError: Data loader neither callable nor valid string constant
        """
        self._print('DataGenerator: _validate_data_loader')
        result = data_loader
        if isinstance(data_loader, str):
            mapping = {
                'numpy': self._data_loader_numpy,
                'numpy_dict': self._data_loader_numpy_dict,
                'json': self._data_loader_json,
                'json+gzip': self._data_loader_json_gzip,
            }
            if data_loader not in mapping:
                raise ValueError('Unknown DataLoader function name!')
            result = mapping[data_loader]
        if not callable(result):
            raise ValueError('DataLoader needs to be a callable!')
        return result

    @staticmethod
    def _data_loader_numpy(filename: str) -> Tuple[int, _TData]:
        """
        Default data loader function for numpy files

        :param filename: Filename to be loaded
        :return: Number of data and data contained within the file
        """
        result = np.load(filename)
        return result.shape[0], result

    @staticmethod
    def _data_loader_numpy_dict(filename: str) -> Tuple[int, _TData]:
        """
        Default data loader function for numpy files containing a dictionary

        :param filename: Filename to be loaded
        :return: Number of data and data contained within the file
        """
        result = np.load(filename)
        if result.shape == ():
            try:
                result = result.item()
            except ValueError:
                pass
        try:
            count = result.shape[0]
        except AttributeError:
            count = len(result)
        return count, result

    @staticmethod
    def _data_loader_json(filename: str) -> Tuple[int, _TData]:
        """
        Default data loader function for json files

        :param filename: Filename to be loaded
        :return: Number of data and data contained within the file
        """
        with open(filename) as file:
            data = json.load(file)
            file.close()
        return len(data), data

    @staticmethod
    def _data_loader_json_gzip(filename: str) -> Tuple[int, _TData]:
        """
        Default data loader function for gzipped json files

        :param filename: Filename to be loaded
        :return: Number of data and data contained within the file
        """
        with gzip.open(filename) as file:
            data = file.read().decode()
            data = json.loads(data)
            file.close()
        return len(data), data

    def _get_valid_testing_path(self) -> str:
        """
        Get a valid path for testing which contains data files

        :return: Path to test for files
        :raises ValueError: No valid path available to check for files
        """
        self._print('DataGenerator: _get_valid_testing_path')
        if self._data:
            path = self._data[0]
        elif self._answer:
            path = self._answer[0]
        elif self._config:
            path = self._config[0]
        else:
            raise ValueError('At least one path to data, answer or config needs to be set!')
        return path

    def _count_files(self, file_limit: int = 0) -> int:
        """
        Scan directories and count the number of files

        :param file_limit: Limit the maximum number of files
        :return: Counted number of files in directory
        """
        self._print('DataGenerator: _count_files')
        file_limit = int(max(0, int(file_limit)))
        path = self._get_valid_testing_path()
        result = 0
        while os.path.isfile(path + self._format.format(result)) is True:
            result += 1
        return result if file_limit == 0 else int(min(result, file_limit))

    def _get_file_size(self) -> int:
        """
        Get the number of data entries in each file

        :return: Number of entries per file
        """
        self._print('DataGenerator: _get_file_size')
        path = self._get_valid_testing_path()
        result, _ = self._data_loader(path + self._format.format(0))
        return result

    def __init__(  # pylint: disable=too-many-arguments
            self,
            file_format: str = None,
            path_data: _TPath = None,
            path_answer: _TPath = None,
            path_config: _TPath = None,
            validation: float = None,
            testing: float = None,
            batch_size: int = None,
            file_limit: int = None,
            file_size: int = None,
            data_loader: _TDataLoader = 'numpy',
            verbose: int = 0,
            nocache: bool = False,
    ) -> None:
        """
        Initialization of the DataGenerator class

        :param file_format: Format of the file names
        :param path_data: Single or list of paths to the data
        :param path_answer: Single or list of paths to the answers
        :param path_config: Single or list of paths to the configs
        :param validation: Fraction of the data used for validation
        :param testing: Fraction of the data used for testing
        :param batch_size: Default batch size of the generator
        :param file_limit: Limit the maximum number of data files used
        :param file_size: Explicitly define the number of entries per data file
        :param data_loader: String name or callable function to load data from files
        :param verbose: Integer verbosity value
        :param nocache: !WARNING! Clear linux cache/buffer after each file (for large dataset)
        :raises ValueError: No training data left, too large validation+testing fraction
        """
        self._verbose = int(verbose)
        self._print('DataGenerator: __init__')
        # Assign file format and path parameters
        self._format = file_format if file_format is not None else 'batch_{0:05d}.npy'
        self._data = self._validate_path(path_data if path_data is not None else [])
        self._answer = self._validate_path(path_answer if path_answer is not None else [])
        self._config = self._validate_path(path_config if path_config is not None else [])
        # Store the training/validation/testing split
        self._validation = float(max(0.0, min(1.0, float(validation) if validation is not None else 0.0)))
        self._testing = float(max(0.0, min(1.0, float(testing) if testing is not None else 0.0)))
        if self._validation + self._testing >= 1.0:
            raise ValueError('Validation and Testing splits contain more than 100% of the data!')
        self._batch_size = int(max(1, batch_size if batch_size is not None else 128))
        # Analyse file count and size
        self._nocache = bool(nocache)
        self._data_loader = self._validate_data_loader(data_loader)
        self._file_count = self._count_files(file_limit if file_limit is not None else 0)
        self._file_size = file_size if file_size is not None else self._get_file_size()
        # Plot status
        if verbose >= 1:
            self._print_summary(verbose)

    def _split_count(self) -> Tuple[int, int, int]:
        """
        Calculate the file based splitting in training, validation, testing

        :return: Tuple of file counts
        """
        self._print('DataGenerator: _split_count')
        train = int(round((1.0 - self._validation - self._testing) * self._file_count))
        if self._validation + self._testing > 0.0:
            remain = self._file_count - train
            validate = int(round((self._validation / (self._validation + self._testing)) * remain))
            test = remain - validate
        else:
            validate = 0
            test = 0
        return train, validate, test

    def _batches_per_file(self, batch_size: int = 1) -> int:
        """
        Calculate batches per file based in given batch size

        :param batch_size: Batch size
        :return: Number of batches per file
        """
        self._print('DataGenerator: _batches_per_file')
        return int(math.ceil(self._file_size / batch_size))

    def _print_summary(self, verbose: int = 1) -> None:
        """
        Print a summary of the DataGenerator setup

        :param verbose: Verbosity level
        """
        self._print('DataGenerator: _print_summary')
        print('Creating new DataGenerator class:')
        if self._nocache is True:
            print('   !!! WARNING !!! Clearing linux cache/buffer is enabled!')
        print('{:20s} = {:s}'.format('File format', self._format))
        for path in self._data:
            print('{:20s} = {:s}'.format('Data path', path))
        for path in self._answer:
            print('{:20s} = {:s}'.format('Answer path', path))
        for path in self._config:
            print('{:20s} = {:s}'.format('Config path', path))
        train, validate, testing = self._split_count()
        print('{:20s} = {:8d}'.format('Number of files', self._file_count))
        if verbose >= 2:
            print('{:20s} = {:8d}'.format('    Training data', train))
            print('{:20s} = {:8d}'.format('    Validation data', validate))
            print('{:20s} = {:8d}'.format('    Testing data', testing))
        print('{:20s} = {:8d}'.format('Entries per file', self._file_size))
        batches_per_file = self._batches_per_file(self._batch_size)
        print('{:20s} = {:8d}'.format('Default batch size', self._batch_size))
        print('{:20s} = {:8d}'.format('Batches per file', batches_per_file))
        if verbose >= 2:
            print('{:20s} = {:8d} entries in {:8d} batches'.format(
                '    Training data',
                train * self._file_size,
                train * batches_per_file
            ))
            print('{:20s} = {:8d} entries in {:8d} batches'.format(
                '    Validation data',
                validate * self._file_size,
                validate * batches_per_file
            ))
            print('{:20s} = {:8d} entries in {:8d} batches'.format(
                '    Testing data',
                testing * self._file_size,
                testing * batches_per_file
            ))

    def _get_file_idxs(self, dataset: str = 'training') -> List[int]:
        """
        Get the indices of the files belonging to given data

        The data argument is a string containing of keywords joined by a
        plus sign. Keywords are: all, training, validation and testing

        :param dataset: String describing data set to be used
        :return: List of resulting file indices
        """
        self._print('DataGenerator: _get_file_idxs')
        result = []  # type: List[int]
        training, validation, testing = self._split_count()
        include_train, include_validate, include_test = False, False, False
        list_train = np.arange(training).tolist()
        list_validate = (np.arange(validation) + training).tolist()
        list_test = (np.arange(testing) + training + validation).tolist()
        parts = dataset.split('+')
        for part in parts:
            if part in ['all', 'train', 'training']:
                include_train = True
            if part in ['all', 'validate', 'validation']:
                include_validate = True
            if part in ['all', 'test', 'testing']:
                include_test = True
        if include_train:
            result = [*result, *list_train]
        if include_validate:
            result = [*result, *list_validate]
        if include_test:
            result = [*result, *list_test]
        return result

    def _shuffle_indices(self, indices: Union[np.ndarray, List[int]], shuffle: bool = True) -> np.ndarray:
        """
        Shuffle a list of indices

        :param indices: List of indices as list or numpy ndarray
        :param shuffle: Whether to shuffle
        :return: Shuffled list of indices
        """
        self._print('DataGenerator: _shuffle_indices')
        if shuffle:
            indices = np.asarray(indices) if isinstance(indices, list) else indices
            np.random.shuffle(indices)
        return np.asarray(indices)

    def _load_files(
            self,
            file_idx: int,
            load_config: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Load all data/answer/config according to given file index

        :param file_idx: File index to load data for
        :param load_config: Whether to include the config data
        :return: Lists of the entry indices, data, answers and config
        """
        self._print('DataGenerator: _load_files')
        result_idx = np.arange(self._file_size) + (file_idx * self._file_size)
        result_data = []  # type: List[np.ndarray]
        result_answer = []  # type: List[np.ndarray]
        result_config = []  # type: List[np.ndarray]
        for path in self._data:
            _, data = self._data_loader(path + self._format.format(file_idx))
            result_data = [*result_data, data]
        for path in self._answer:
            _, answer = self._data_loader(path + self._format.format(file_idx))
            result_answer = [*result_answer, answer]
        if load_config:
            for path in self._config:
                _, config = self._data_loader(path + self._format.format(file_idx))
                result_config = [*result_config, config]
        return result_idx, result_data, result_answer, result_config

    def _validate_enhancements(self, enhancements: _TEnhancement) -> _TEnhancementList:
        """
        Validate enhancements and return as list

        :param enhancements: Enhancements (single or list)
        :return: List of enhancements
        """
        self._print('DataGenerator: _validate_enhancements')
        result = []  # type: _TEnhancementList
        if enhancements is not None:
            result = list(enhancements) if isinstance(enhancements, (list, tuple)) else [enhancements, ]
        return result

    def _validate_selections(self, selections: _TSelection) -> _TSelectionList:
        """
        Validate selections and return as list

        :param selections: Selections (single or list)
        :return: List of selections
        """
        self._print('DataGenerator: _validate_selections')
        result = []  # type: _TSelectionList
        if selections is not None:
            result = list(selections) if isinstance(selections, (list, tuple)) else [selections, ]
        return result

    def _apply_enhancements(
            self,
            enhancements: _TEnhancementList,
            data: _TDataList,
            answer: _TDataList,
            config: _TDataList
    ) -> Tuple[_TDataList, _TDataList, _TDataList]:
        """
        Apply data enhancement functions

        :param enhancements: List of enhancement functions
        :param data: Data to be enhanced
        :param answer: Answers to be enhanced
        :param config: Config to be enhanced
        :return: Tuple of enhanced data, answers, config
        """
        self._print('DataGenerator: _apply_enhancements')
        for enhancement in enhancements:
            data, answer, config = enhancement(data, answer, config)
        return data, answer, config

    def _apply_selection_vector(self, data: _TData, vector: np.ndarray) -> _TData:
        """
        Apply a selection vector to one data entry (or each element if dictionary)

        :param data: Data entry to apply selection on
        :param vector: Selection vector
        :return: Reduced data (or dictionary of data)
        """
        if isinstance(data, dict):
            result = {}
            for key in data.keys():
                result[key] = self._apply_selection_vector(data[key], vector)
        elif isinstance(data, list):
            result = np.asarray(data)[vector]
        else:
            result = data[vector]
        return result

    def _apply_selections(
            self,
            selections: _TSelectionList,
            idx: np.ndarray,
            data: _TDataList,
            answer: _TDataList,
            config: _TDataList
    ) -> Tuple[np.ndarray, _TDataList, _TDataList, _TDataList]:
        """
        Apply data selection functions

        :param selections: List of data selection functions
        :param idx: Indices of the data entries
        :param data: Data to be selected
        :param answer: Answers to be selected
        :param config: Config to be selected
        :return: Tuple of selected data entry indices, data, answers, config
        """
        self._print('DataGenerator: _apply_selections')
        for selection in selections:
            vector = selection(idx, data, answer, config)
            idx = idx[vector]
            for i, _ in enumerate(data):
                data[i] = self._apply_selection_vector(data[i], vector)
            for i, _ in enumerate(answer):
                answer[i] = self._apply_selection_vector(answer[i], vector)
            for i, _ in enumerate(config):
                config[i] = self._apply_selection_vector(config[i], vector)
        return idx, data, answer, config

    def _delistify(self, data: _TDataList) -> Union[_TData, _TDataList]:
        """
        Return the single value in case of length one lists

        :param data: List to be turn into single value if needed
        :return: List or single value
        """
        self._print('DataGenerator: _delistify')
        return data[0] if len(data) == 1 else data  # type: ignore

    def _apply_slicing(self, data: _TData, slice_start: int = 0, slice_end: int = -1) -> _TData:
        """
        Apply array-like slicing to data (including dictionaries)

        :param data: Data to be sliced
        :param slice_start: Start index
        :param slice_end: End index
        :return: Sliced data
        """
        if isinstance(data, dict):
            result = {}
            for key in data.keys():
                result[key] = self._apply_slicing(data[key], slice_start, slice_end)
        else:
            result = data[slice_start:slice_end]
        return result

    # noinspection SpellCheckingInspection
    @staticmethod
    def clear_cache() -> None:
        """
        !!! WARNING !!! Use only when certain internal about functionality

        Clears the linux cache/buffer for preventing OOM errors on large dataset.

        The actual command to be executed requires SUDO access, and thus potentially
        a password prompt. In order to circumvent this problem our training
        environments contain the /usr/local/sbin/clearcache.sh shell script that
        wraps the commands and is authorized to be SUDOed from anyone without
        password prompt.
        """
        if os.path.isfile('/usr/local/sbin/clearcache.sh'):
            os.system('sudo /usr/local/sbin/clearcache.sh')
        else:
            os.system('sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches"')

    def _clear_cache(self) -> None:
        """
        !!! WARNING !!! Use only when certain internal about functionality

        Clears the linux cache/buffer for preventing OOM errors on large dataset
        """
        if self._nocache is True:
            self.clear_cache()

    def batches(self, dataset: str = 'training', batch_size: int = None) -> int:
        """
        Get the number of batches for a certain data set

        :param dataset: String describing data set to be used
        :param batch_size: Batch size to use
        :return: Number of according batches
        """
        self._print('DataGenerator: batches')
        idx = self._get_file_idxs(dataset)
        batch_size = batch_size if batch_size is not None else self._batch_size
        file_batches = int(math.ceil(self._file_size / batch_size))
        return file_batches * len(idx)

    def count(
            self,
            dataset: str = 'training',
            preparation: _TEnhancement = None,
            selection: _TSelection = None,
            categorical_count: bool = False
    ) -> Union[int, Tuple[int, Any]]:
        """
        Count the number of valid entries

        :param dataset: String describing data set to be used
        :param preparation: Preparation enhancement functions to be applied
        :param selection: Selection functions to be applied
        :param categorical_count: Interpret answer as categorical and count them
        :return: Number of resulting data entries
        """
        self._print('DataGenerator: count')
        result = 0
        preparation = self._validate_enhancements(preparation)
        selection = self._validate_selections(selection)
        idxs = self._get_file_idxs(dataset)
        counting = None
        for idx in idxs:
            self._clear_cache()
            entry_idx, data, answer, config = self._load_files(idx, True)
            data, answer, config = self._apply_enhancements(preparation, data, answer, config)
            entry_idx, data, answer, config = self._apply_selections(selection, entry_idx, data, answer, config)
            if categorical_count:
                tmp_counting = np.asarray(answer)
                while len(tmp_counting.shape) > 1:
                    tmp_counting = np.sum(tmp_counting, axis=0)
                counting = tmp_counting if counting is None else counting + tmp_counting
            result += len(entry_idx)
        if categorical_count:
            return result, counting.astype(int)
        return result

    def generator(  # pylint: disable=too-many-arguments,too-many-locals,missing-yield-doc
            self,
            dataset: str = 'training',
            batch_size: int = None,
            shuffle: bool = True,
            prepend_idx: bool = False,
            append_config: bool = False,
            preparation: _TEnhancement = None,
            selection: _TSelection = None,
            enhancement: _TEnhancement = None
    ) -> Generator[Tuple[Union[_TData, _TDataList], ...], None, None]:
        """
        Generator method to provide batches of data/answer/configs

        :param dataset: String describing data set to be used
        :param batch_size: Batch size to be used
        :param shuffle: Shuffle files and batches within the files
        :param prepend_idx: Prepend an data entry index to the result
        :param append_config: Append the config data to the result
        :param preparation: Preparations to be applied
        :param selection: Selections to be applied
        :param enhancement: Enhancements to be applied
        :yields: Tuple of data/answers, optionally also index and config
        """
        self._print('DataGenerator: generator')

        # Prepare input arguments
        preparation = self._validate_enhancements(preparation)
        selection = self._validate_selections(selection)
        enhancement = self._validate_enhancements(enhancement)
        file_idxs = self._get_file_idxs(dataset)
        batch_size = batch_size if batch_size is not None else self._batch_size
        file_batches = int(math.ceil(self._file_size / batch_size))
        load_config = append_config or (len(preparation) > 0) or (len(selection) > 0) or (len(enhancement) > 0)

        # Loop the generator over all according files
        while True:
            file_idxs = self._shuffle_indices(file_idxs, shuffle).tolist()
            for file_idx in file_idxs:
                self._clear_cache()
                idxs, values, answers, configs = self._load_files(file_idx, load_config)

                # Apply the preparation
                values, answers, configs = self._apply_enhancements(
                    preparation,
                    values,
                    answers,
                    configs
                )

                # Loop over all batches of the file
                batch_idxs = np.arange(file_batches).tolist()
                # noinspection PyTypeChecker
                batch_idxs = self._shuffle_indices(batch_idxs, shuffle).tolist()
                for batch_idx in batch_idxs:

                    # Get the batch data
                    batch_start = int(max(0, batch_idx * batch_size))
                    batch_end = int(min(self._file_size, (batch_idx + 1) * batch_size))
                    result_idxs = idxs[batch_start:batch_end]
                    result_data = []  # type: List[np.ndarray]
                    for value in values:
                        result_data = [*result_data, self._apply_slicing(value, batch_start, batch_end)]
                    result_answer = []  # type: List[np.ndarray]
                    for answer in answers:
                        result_answer = [*result_answer, self._apply_slicing(answer, batch_start, batch_end)]
                    result_config = []  # type: List[np.ndarray]
                    for config in configs:
                        result_config = [*result_config, self._apply_slicing(config, batch_start, batch_end)]

                    # Apply the selection
                    result_idxs, result_data, result_answer, result_config = self._apply_selections(
                        selection,
                        result_idxs,
                        result_data,
                        result_answer,
                        result_config
                    )

                    # Apply the enhancements
                    result_data, result_answer, result_config = self._apply_enhancements(
                        enhancement,
                        result_data,
                        result_answer,
                        result_config
                    )

                    # Build the result
                    result = (
                        self._delistify(result_data),
                        self._delistify(result_answer)
                    )  # type: Tuple[Union[_TData, _TDataList], ...]
                    if prepend_idx:
                        result = (result_idxs, *result)
                    if append_config:
                        result = (*result, self._delistify(result_config))
                    yield result
