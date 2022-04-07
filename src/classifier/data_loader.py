from tensorflow.keras.utils import Sequence
from src import static
import multiprocessing as mp
import os
import numpy as np
from typing import Optional, Tuple
import glob
import math
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)


val_datagen = ImageDataGenerator()


def rescaler_parts(_min, _max):
    return 1 / (_max - _min), -(_min) / (_max - _min)


def get_rescalers(shape):
    multipliers = np.ones(shape=(*shape, len(static.data_names_ordered)))
    additives = np.ones(shape=(*shape, len(static.data_names_ordered)))
    for i, name in enumerate(static.data_names_ordered):
        multiplier, additive = rescaler_parts(*static.rescale_min_max[name])
        multipliers[..., i] = multiplier 
        additives[..., i] = additive 
    return multipliers, additives


class XviewSequence(Sequence):
    def __init__(self, directory: str, batch_size: int = 64,
                 channels_ix: Tuple[int] = (static.VV, static.VH, static.BATHYMETRY),
                 as_bool: bool = False, image_augmenter: ImageDataGenerator = ImageDataGenerator(),
                 scene_id: Optional[str] = None, train: bool = True):
        if scene_id is None:
            scene_id = '*'
        self.x_paths = sorted(glob.glob(os.path.join(directory, scene_id, 'X_[0-9]*.npy')))
        self.y_paths = sorted(glob.glob(os.path.join(directory, scene_id, 'y_[0-9]*.npy')))
        self.data_augmentater = image_augmenter
        self.train = train
        self.asbool = as_bool
        self.channels_ix = channels_ix
        self.batch_size=batch_size
        n = len(self.x_paths)
        if n != len(self.y_paths):
            raise Exception(f"Datasize inconsistent. X is {n}, y is {len(self.y_paths)}")
        rescalers = get_rescalers(self.input_data_shape[:-1])
        self.rescale_multiply = rescalers[0][..., self.channels_ix]
        self.rescale_add = rescalers[1][..., self.channels_ix]
    
    def _rescale(self, X):
        X = X * self.rescale_multiply + self.rescale_add
        X[X < 0 ] = 0 # in case
        X[X > 1 ] = 1
        return X
    
    def __len__(self):
        return math.ceil(len(self.x_paths) / self.batch_size)
       
    def on_epoch_end(self):
        combined = list(zip(self.x_paths, self.y_paths))
        shuffle(combined)
        self.x_paths, self.y_paths = zip(*combined)

    def __getitem__(self, ix):
        Xs = []
        ys = []
        for x_path, y_path in zip(self.x_paths[ix*self.batch_size:(ix+1)*self.batch_size],
                                  self.y_paths[ix*self.batch_size:(ix+1)*self.batch_size]):
            X = self._rescale(np.load(x_path)[..., self.channels_ix])
            if self.train:
                X = self.data_augmentater.random_transform(X)
            Xs.append(X)
            ys.append(np.load(y_path))
        X = np.stack(Xs)
        y = np.stack(ys)
        if self.asbool:
            y = y != 0 # True if Something, False if Nothing
            
        return X, y
    
    def _labels(self):
        return np.hstack(list(map(np.load, self.y_paths)))
    
    def _features(self):
        return np.vstack(list(map(np.load, self.x_paths)))[..., self.channels_ix]
    
    @property
    def input_data_shape(self):
        return np.load(self.x_paths[0])[..., self.channels_ix].shape
    
    @property
    def shape(self):
        return (len(self.x_paths), *self.input_data_shape)



def multiprocess_load(paths, processes=None):
    with mp.Pool(processes=processes) as pool:
        data_list = pool.map(np.load, paths)
    data = np.stack(data_list)
    return data


class XviewSequenceInMemory(Sequence):
    def __init__(self, directory: str, batch_size: int = 64,
                 channels_ix: Tuple[int] = (static.VV, static.VH, static.BATHYMETRY),
                 as_bool: bool = False, image_augmenter: ImageDataGenerator = ImageDataGenerator(),
                 scene_id: Optional[str] = None, train: bool = True, processes: Optional[int]=None):
        if scene_id is None:
            scene_id = '*'
        self.x_paths = sorted(glob.glob(os.path.join(directory, scene_id, 'X_[0-9]*.npy')))
        self.y_paths = sorted(glob.glob(os.path.join(directory, scene_id, 'y_[0-9]*.npy')))
        self.data_augmentater = image_augmenter
        self.train = train
        self.asbool = as_bool
        self.channels_ix = channels_ix
        self.batch_size = batch_size
        n = len(self.x_paths)
        if n != len(self.y_paths):
            raise Exception("Datasize bad")
        rescalers = get_rescalers(self.input_data_shape[:-1])
        self.rescale_multiply = rescalers[0][..., self.channels_ix]
        self.rescale_add = rescalers[1][..., self.channels_ix]
        print(f'Loading In Memory: {len(self.x_paths)}')

        self.y = multiprocess_load(self.y_paths, processes)
        self.X = multiprocess_load(self.x_paths, processes)[..., self.channels_ix] * self.rescale_multiply + self.rescale_add
        self.indexes = np.arange(self.X.shape[0])

        self.X[self.X < 0] = 0
        self.X[self.X > 1] = 1

    def __len__(self):
        return math.ceil(len(self.x_paths) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, ix):
        ix = self.indexes[ix * self.batch_size: (ix + 1) * self.batch_size]
        y = self.y[ix]
        if self.train:
            X = np.stack([self.data_augmentater.random_transform(self.X[i, ...]) for i in ix])
        else:
            X = self.X[ix, ...]
        return X, y

    def _labels(self):
        return self.y

    def _features(self):
        return self.X

    @property
    def input_data_shape(self):
        return np.load(self.x_paths[0])[..., self.channels_ix].shape

    @property
    def shape(self):
        return (len(self.x_paths), *self.input_data_shape)


class XviewDataset():
    # DEPRECATED
    def __init__(self, directory, batch_size=64):
        self.x_paths = sorted(glob.glob(os.path.join(directory,'*', 'X_[0-9]*.npy')))
        self.y_paths = sorted(glob.glob(os.path.join(directory,'*', 'y_[0-9]*.npy')))
        self.metadata_paths = sorted(glob.glob(os.path.join(directory,'*', 'metadata_[0-9]*.csv')))
        self.batch_size=batch_size
        self.ix = 0
        n = len(self.x_paths)
        if (n != len(self.y_paths)) or (n != len(self.metadata_paths)):
            raise Exception("Datasize bad")

    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        self.ix = 0
        return self
    
    def __len__(self):
        return len(self.x_paths)
    
    @property
    def steps_per_epoch(self):
        return int(self.__len__() // self.batch_size)
    
    def __next__(self):
        self.ix += self.batch_size
        if self.ix > self.__len__(): 
            self.reset_epoch()
#             raise StopIteration
        return self.__getitem__(self.ix)
    
    def reset_epoch(self):
        print('reset')
        self.ix = 0
        combined = list(zip(self.x_paths,self.y_paths, self.metadata_paths))
        shuffle(combined)
        self.x_paths, self.y_paths, self.metadata_paths = zip(*combined)

    

    def __getitem__(self, ix):
        Xs = []
        ys = []
        metadatas = []
        for x_path, y_path in zip(self.x_paths[ix:ix+self.batch_size],
                                  self.y_paths[ix:ix+self.batch_size]):
            Xs.append(np.load(x_path))
            ys.append(np.load(y_path))
        X = np.stack(Xs)
        y = np.stack(ys)
        return X, y


class XviewLength(XviewSequence):
    def __init__(self, directory: str, batch_size: int = 64,
                 channels_ix: Tuple[int] = (static.VV, static.VH, static.BATHYMETRY),
                 as_bool: bool = False, image_augmenter: ImageDataGenerator = ImageDataGenerator(),
                 scene_id: Optional[str] = None, train: bool = True):
        if scene_id is None:
            scene_id = '*'

        self.x_paths = sorted(glob.glob(os.path.join(directory, scene_id, 'X_[0-9]*.npy')))
        self.y_paths = sorted(glob.glob(os.path.join(directory, scene_id, 'metadata_[0-9]*.csv')))
        self.mask_valid = self._valid_data_mask()
        self.x_paths = [sample[0] for sample in zip(self.x_paths, self.mask_valid) if sample[1]]
        self.y_paths = [sample[0] for sample in zip(self.y_paths, self.mask_valid) if sample[1]]

        self.data_augmentater = image_augmenter
        self.train = train
        self.asbool = as_bool
        self.channels_ix = channels_ix
        self.batch_size = batch_size
        n = len(self.x_paths)
        if n != len(self.y_paths):
            raise Exception(f"Datasize inconsistent. X is {n}, y is {len(self.y_paths)}")
        rescalers = get_rescalers(self.input_data_shape[:-1])
        self.rescale_multiply = rescalers[0][..., self.channels_ix]
        self.rescale_add = rescalers[1][..., self.channels_ix]

    def _valid_data_mask(self):
        return [length is not np.nan for length in list(map(self._y_length, self.y_paths))]

    def _rescale(self, X):
        X = X * self.rescale_multiply + self.rescale_add
        X[X < 0] = 0  # in case
        X[X > 1] = 1
        return X

    def __len__(self):
        return math.ceil(len(self.x_paths) / self.batch_size)

    def on_epoch_end(self):
        combined = list(zip(self.x_paths, self.y_paths))
        shuffle(combined)
        self.x_paths, self.y_paths = zip(*combined)

    def __getitem__(self, ix):
        Xs = []
        ys = []
        for x_path, y_path in zip(self.x_paths[ix * self.batch_size:(ix + 1) * self.batch_size],
                                  self.y_paths[ix * self.batch_size:(ix + 1) * self.batch_size]):
            X = self._rescale(np.load(x_path)[..., self.channels_ix])
            if self.train:
                X = self.data_augmentater.random_transform(X)
            Xs.append(X)
            # ys.append(np.load(y_path))
            ys.append(self._y_length(y_path))
        X = np.stack(Xs)
        y = np.stack(ys)
        if self.asbool:
            y = y != 0  # True if Something, False if Nothing

        return X, y

    def _labels(self):
        # use functools.partial to use named arguments in map()
        # mapfunc = partial(pd.read_csv, index_col=0)
        # return np.hstack([length_m.transpose()['vessel_length_m'][0] for length_m in map(mapfunc, self.y_paths)])

        # use class function (more useful)
        return np.hstack(list(map(self._y_length, self.y_paths)))

    def _y_length(self, y_path):
        # return float(pd.read_csv(y_path, index_col=0).transpose()['vessel_length_m'][0])

        with open(y_path, 'r') as f:
            for line in range(4):
                length_data = f.readline()
            length_data = length_data.split(',')[1]

        if length_data.strip():
            return float(length_data)

        return np.nan

    def _features(self):
        return np.vstack(list(map(np.load, self.x_paths)))[..., self.channels_ix]

    @property
    def input_data_shape(self):
        return np.load(self.x_paths[0])[..., self.channels_ix].shape

    @property
    def shape(self):
        return (len(self.x_paths), *self.input_data_shape)


