import os
import sys
import math
import json
import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Dense, Lambda
from keras.layers import Activation, Flatten, Dropout, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


BASE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/data'
IMG_SHP = (66, 200, 3)


class ImageProcessor(object):

    @classmethod
    def preprocess_image(cls, image):
        """
        Preprocess image:
        1. Crop image: remove unrelated protion of the image
                       (e.g. hood, everything above the road surface)
        2. Scale image: to reduce calculation
        3. Change color space
        """
        image = image[60:145, 40:280]

        image = cv2.resize(image, (IMG_SHP[1], IMG_SHP[0]), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        return image

    @classmethod
    def augment_data(cls, img, y):
        """
        https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
        """
        img, y = cls.shift_image(img, y)
        img, y = cls.flip_image(img, y)

        img = cls.change_brightness(img)
        img = cls.add_shadow(img)

        return img, y

    @classmethod
    def choose_image(cls, record, y, delta=.27, idx=None):
        """
        Side cameras : simulate driving off center
        """
        delta = [delta, 0, -delta]
        image = [record.left_image,
                 record.center_image,
                 record.right_image]

        if idx is None:
            idx = np.random.randint(3)
        return image[idx], y + delta[idx]

    @classmethod
    def perturb_angle(cls, angle):
        """
        https://medium.com/@acflippo/cloning-driving-behavior-by-augmenting-steering-angles-5faf7ea8a125#.9qo2nhv3o
        """
        delta = 0.01
        return angle * np.random.uniform(1-delta, 1+delta)

    @classmethod
    def shift_image(cls, img, y):
        """
        Shift: simulate different position on the road, up hill, down hill
        """
        if np.random.uniform() > .25:
            return img, y

        max_dx, max_dy = 50, 20
        dx = np.random.uniform(-max_dx, max_dx)
        dy = np.random.uniform(-max_dy, max_dy)
        y += dx / max_dx * .2

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return img, y

    @classmethod
    def flip_image(clsf, img, y, random=True):
        """
        Flip: simulate different side (direction) of the road
        """
        if not random or np.random.randint(2):
            img = cv2.flip(img, 1)
            y = -y
        return img, y

    @classmethod
    def change_brightness(cls, img, factor=None):
        """
        Brightness: simulate different time in the day
        """
        factor = factor or np.random.uniform(0.15, 1.2)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,2] = (img[:,:,2] * factor).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img

    @classmethod
    def add_shadow(cls, img):
        """
        Shadow: simulate shadow cast on the road
        """
        rows, cols, _ = img.shape
        top_x = 0
        top_y = np.random.uniform(cols)

        bot_x = rows
        bot_y = np.random.uniform(cols)

        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        X_m, Y_m = np.mgrid[0:rows,0:cols]
        mask = np.zeros_like(img[:,:,1])
        mask[(  X_m - top_x)*(bot_y - top_y) -
             (bot_x - top_x)*(  Y_m - top_y) >= 0] = 1

        bright = np.random.uniform(0.15, 0.95)
        mask = (mask == np.random.randint(2))
        img[:,:,1][mask] = img[:,:,1][mask] * bright

        img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img


# Dataset definitions
class ImageRecord(object):
    """ A data structure to load and hold image record. """

    IMG_PATH = BASE_PATH

    def __init__(self, meta):

        self.steering = float(meta.get('steering') or 0)
        self.throttle = float(meta.get('throttle') or 0)
        self.speed    = float(meta.get('speed') or 0)
        self.brake    = float(meta.get('brake') or 0)

        self.left_path   = os.path.join(self.IMG_PATH,
                                        meta.get('left').strip())
        self.center_path = os.path.join(self.IMG_PATH,
                                        meta.get('center').strip())
        self.right_path  = os.path.join(self.IMG_PATH,
                                        meta.get('right').strip())

        self._left_image   = None
        self._center_image = None
        self._right_image  = None

    @property
    def left_image(self):
        if self._left_image is None:
            self._left_image = self.load_image(self.left_path)
        return self._left_image

    @property
    def center_image(self):
        if self._center_image is None:
            self._center_image = self.load_image(self.center_path)
        return self._center_image

    @property
    def right_image(self):
        if self._right_image is None:
            self._right_image = self.load_image(self.right_path)
        return self._right_image

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            print('Failed to load image : {}'.format(path))
        return image

    def __str__(self):
        output = ''
        output += 'Steering: {}\n'.format(self.steering)
        output += 'Throttle: {}\n'.format(self.throttle)
        output += 'Speed: {}\n'.format(self.speed)
        output += 'Brake: {}\n'.format(self.brake)
        output += '\n'
        output += 'Left camera image path: {}\n'.format(self.left_path)
        output += 'Center camera image path: {}\n'.format(self.center_path)
        output += 'Right camera image path: {}\n'.format(self.right_path)
        output += '\n\n'
        return output


class BaseImageDataset(object):
    """ A data structure to hold a set of records. """

    def __init__(self, test_size=0.1):

        self.SIZE_SCALE = 1

        self.X_train = None
        self.y_train = None
        self._X_train = None
        self._y_train = None
        self.train_size = 0

        self.X_validate = None
        self.y_validate = None
        self._X_validate = None
        self._y_validate = None
        self.validation_size = 0

        self._test_size = test_size

        self.load_data()

    @property
    def log_path(self):
        if not hasattr(self, 'LOG_PATH'):
            raise NotImplementedError

        return self.LOG_PATH

    @property
    def test_size(self):
        return self._test_size

    @property
    def image_shape(self):
        if self._X_train is None:
            return None
        return self._X_train[0].center_image.shape

    def load_data(self):
        """
        """
        df = pd.read_csv(self.LOG_PATH)

        X_train = []
        y_train = []

        # load data from log
        for _, data in df.iterrows():
            record = ImageRecord(data)
            X_train.append(record)
            y_train.append(record.steering)

        # split data
        X_train, X_validate, y_train, y_validate = train_test_split(
                X_train, y_train, test_size=self.test_size)

        self._X_train = X_train
        self._y_train = y_train
        self.train_size = len(X_train) * self.SIZE_SCALE

        self._X_validate = X_validate
        self._y_validate = y_validate
        self.validation_size = len(X_validate) * self.SIZE_SCALE


    ##########################################################################
    ## Data
    ##########################################################################
    @property
    def train_data(self):
        if self.X_train is None or self.y_train is None:
            self.X_train, self.y_train = \
                    self.process_data(self._X_train, self._y_train)
        return self.X_train, self.y_train

    @property
    def validation_data(self):
        if self.X_validate is None or self.y_validate is None:
            self.X_validate, self.y_validate = \
                    self.process_data(self._X_validate, self._y_validate)
        return self.X_validate, self.y_validate

    def process_data(self, raw_X, raw_y, skip_rate=0.2, center_only=False):
        X, y = [], []

        i_range = [1] if center_only else range(3)

        for rec, out in zip(raw_X, raw_y):
            for i in i_range:
                img, new_out = ImageProcessor.choose_image(rec, out, idx=i)
                img          = ImageProcessor.preprocess_image(img)
                img, new_out = ImageProcessor.augment_data(img, new_out)

                if abs(new_out) > 0.1 or np.random.uniform() > skip_rate:
                    X.append(img)
                    y.append(new_out)

                if abs(new_out) > 0.1:
                    img, new_out = ImageProcessor.flip_image(img, new_out, random=False)
                    X.append(img)
                    y.append(new_out)

        #shuffled_indexes = np.arange(len(X))
        #shffuled_indexes = np.random.shuffle(shuffled_indexes)
        #return np.array(X)[shuffled_indexes], np.array(y)[shuffled_indexes]
        return np.array(X), np.array(y)

    ##########################################################################
    ## Data generator
    ##########################################################################
    @property
    def train_data_generator(self):
        return self.data_generator(self._X_train, self._y_train, skip_rate=0)

    @property
    def validation_data_generator(self):
        return self.data_generator(self._X_validate, self._y_validate, skip_rate=0)

    def data_generator(self, X, y, batch_size=128, skip_rate=.8):
        """
        Return a generator that go through all records
        and their augmented data
        """
        nb_batches = math.ceil(self.train_size/batch_size)
        batch = 0
        idx = 0

        while True:

            X_batch = []
            y_batch = []

            while len(X_batch) < batch_size:

                img, out = ImageProcessor.choose_image(X[idx], y[idx])
                img      = ImageProcessor.preprocess_image(img)
                img, out = ImageProcessor.augment_data(img, out)

                if abs(out) > 0.1 or np.random.uniform() > skip_rate:
                    X_batch.append(img)
                    y_batch.append(out)

                idx += 1
                idx %= len(X)

            yield np.array(X_batch), np.array(y_batch)

            batch += 1
            if batch == nb_batches:
                batch = 0
                skip_rate /= 2


class UdacityDataset(BaseImageDataset):

    LOG_PATH = os.path.join(BASE_PATH, 'driving_log.csv')


# Network definitions
class NetworkModel(object):
    """ """

    def __init__(self, init='he_normal', activation='elu', dropout=0.5,
            batch_size=256):

        self.init = init
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size

        self.loss = 'mse'
        self.optimizer = Adam(lr=0.001)
        self.metrics = ['accuracy']

        self._model = None
        self._weights = None

        cls_name = self.__class__.__name__
        self._model_filename = '{}.json'.format(cls_name.lower())
        self._weights_filename = '{}.h5'.format(cls_name.lower())

    @property
    def model(self):
        raise NotImplementedError

    def _get_input_shape(self):
        """ """
        return IMG_SHP

    def fit(self, X_train, y_train, X_validate, y_validate, nb_epoch):
        early_stop = EarlyStopping(monitor='val_loss',
                min_delta=0.0001, patience=5)
        checkpoint = ModelCheckpoint(self._weights_filename,
                monitor='val_loss', save_best_only=True)

        callbacks = []
        callbacks.append(early_stop)
        callbacks.append(checkpoint)

        self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            validation_data=(X_validate, y_validate),
            nb_epoch=nb_epoch,
            callbacks=callbacks
        )

    def fit_generator(self, train_gen, val_gen, nb_epoch,
            samples_per_epoch, nb_val_samples):

        batch_size = self.batch_size
        samples_per_epoch = math.ceil(samples_per_epoch/batch_size) * batch_size
        nb_val_samples = math.ceil(nb_val_samples/batch_size) * batch_size

        early_stop = EarlyStopping(monitor='val_loss',
                min_delta=0.0001, patience=3)
        checkpoint = ModelCheckpoint(self._weights_filename,
                monitor='val_loss', save_best_only=True)

        callbacks = []
        callbacks.append(early_stop)
        callbacks.append(checkpoint)

        self.model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            nb_epoch=nb_epoch,
            samples_per_epoch=samples_per_epoch,
            nb_val_samples=nb_val_samples,
            callbacks=callbacks
        )

    def predict(self, imgs):
        return self.model.predict(imgs)

    def save_model(self, name_prefix=''):
        with open(name_prefix + self._model_filename, 'w') as f:
            json.dump(self.model.to_json(), f)

        print('Model saved. {}'.format(self._model_filename))

    def save(self, name_prefix=''):
        with open(name_prefix + self._model_filename, 'w') as f:
            json.dump(self.model.to_json(), f)
        self.model.save_weights(name_prefix + self._weights_filename)

        print('Model saved. {}, {}'.format(
            self._model_filename, self._weights_filename))

    def restore(self, name_prefix=''):
        self._model = None
        model_json = {}

        with open(name_prefix + self._model_filename, 'r') as f:
            model_json = json.load(f)

        self._model = model_from_json(model_json)
        self.model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['mse'])
        self.model.load_weights(name_prefix + self._weights_filename)
        print('Model loaded.')

    def __str__(self):
        if self.model:
            return self.model.summary()
        return ''


class NvidiaNet(NetworkModel):
    """ """

    @property
    def model(self):

        if self._model is not None:
            return self._model

        input_shape = self._get_input_shape()

        kwargs = {
            'border_mode' : 'valid',
            'init'        : self.init,
            'activation'  : self.activation
        }

        model = Sequential()
        # nomalization
        model.add(Lambda(lambda x: x/127.5 - 1,
            input_shape=input_shape))

        #model.add(Convolution2D( 3, 1, 1, subsample=(1, 1), **kwargs))

        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), **kwargs))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), **kwargs))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), **kwargs))
        model.add(Convolution2D(64, 3, 3, **kwargs))
        model.add(Convolution2D(64, 3, 3, **kwargs))
        model.add(Flatten())
        model.add(Dropout(self.dropout))
        del kwargs['border_mode']
        model.add(Dense(1164, **kwargs))
        model.add(Dropout(self.dropout))
        model.add(Dense(100, **kwargs))
        model.add(Dense(50, **kwargs))
        model.add(Dense(10, **kwargs))
        model.add(Dense(1))

        model.summary()
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['mse'])

        self._model = model
        return self._model


class CommaAiNet(NetworkModel):
    """
    https://github.com/commaai/research/blob/master/train_steering_model.py
    """
    @property
    def model(self):
        if self._model is not None:
            return self._model

        input_shape = self._get_input_shape()
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(self.dropout))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(self.dropout))
        model.add(ELU())
        model.add(Dense(1))

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['mse'])

        self._model = model
        return self._model


class ChihChuanNet(NetworkModel):
    """ """

    @property
    def model(self):
        if self._model is not None:
            return self._model

        input_shape = self._get_input_shape()

        conv_args = {
            'border_mode' : 'valid',
            'init'        : self.init,
            'activation'  : self.activation
        }

        maxpool_args = {
            'pool_size' : (2, 2),
            'strides'   : (1, 1)
        }

        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1,
            input_shape=input_shape))

        model.add(Convolution2D(3, 1, 1, **conv_args))
        model.add(Convolution2D(32, 3, 3, subsample=(2, 2), **conv_args))
        model.add(MaxPooling2D(**maxpool_args))
        model.add(Dropout(self.dropout))

        model.add(Convolution2D(64, 3, 3, subsample=(2, 2), **conv_args))
        model.add(MaxPooling2D(**maxpool_args))
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), **conv_args))
        model.add(MaxPooling2D(**maxpool_args))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dropout(self.dropout))

        del conv_args['border_mode']
        model.add(Dense(100, **conv_args))
        model.add(Dense(50, **conv_args))
        model.add(Dense(1))

        model.summary()
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['mse'])

        self._model = model
        return self._model


# start application
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--nb_epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--model', type=str, default='NvidiaNet', help='Training model')
    parser.add_argument('--dataset', type=str, default='udacity', help='Dataset')
    parser.add_argument('--load_model', action='store_true', default=False, help='Load model')
    parser.add_argument('--use_generator', action='store_true', default=False, help='Use generator')
    args = parser.parse_args()

    model_names = {
        'NvidiaNet' : NvidiaNet,
        'ChihNet'   : ChihChuanNet,
        'CommaAiNet': CommaAiNet,
    }

    dataset_names = {
        'udacity' : UdacityDataset
    }

    print('Application started...')
    model_cls = model_names.get(args.model)
    dataset_cls = dataset_names.get(args.dataset)

    model = model_cls(dropout=args.dropout, batch_size=args.batch)
    dataset = dataset_cls()

    if args.load_model:
        model.restore()
        model.model.summary()

    # train
    print('Start training...')
    if args.use_generator:
        model.fit_generator(
            train_gen=dataset.train_data_generator,
            val_gen=dataset.validation_data_generator,
            nb_epoch=args.nb_epoch,
            samples_per_epoch=dataset.train_size,
            nb_val_samples=dataset.validation_size
        )
    else:
        model.fit(
            dataset.train_data[0],
            dataset.train_data[1],
            dataset.validation_data[0],
            dataset.validation_data[1],
            args.nb_epoch)

    model.save_model()

    # predict
    #img = dataset.X_train[0].center_image
    #img = np.array([img])
    #pred = model.predict(img)

    backend.clear_session()

