import numpy as np
import datetime

from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.preprocessing import image

from datagen import gen_data
from utils import generate_random_gap

LOAD_WEIGHTS = False
ITERATIONS = 50000
BATCH_SIZE = 4
SEED = 1
IMG_SHAPE = (352, 352, 1)
IMG_HEIGHT, IMG_WIDTH, IMG_CHAN = IMG_SHAPE
DATA_TYPE = 'DATA_GEN'


def build_model():
    input_tensor = Input((None, None, 1))

    x = Conv2D(24, 5, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, strides=1, padding='same', )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(8, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(4, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(2, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output_tensor = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    return Model(input_tensor, output_tensor)


def data_generator(type='DATA_GEN'):
    '''
    Generate data in specific type.
    DATA_GEN: generate data with random graphics, use small disc as gap, non-meaningful data
    DATA_GAP: generate data use small disc as gap on user line-drawings, meaningful data
    DATA_THIN: directly read offline data generated using normalization(thinning)

    :param type: DATA_GEN, DATA_GAP, DATA_THIN
    :return: x_data, y_data
    '''
    # Use both 352 and 176 could achieve better performance
    gap_configs352 = [
        [50, 600, 2, 8, 0, 1],
        [50, 600, 2, 10, 0, 2],
        [1, 2, 5, 15, 0, 3]
    ]

    # gap_configs176 = [
    #     [50, 200, 1, 4, 0, 1],
    #     [50, 200, 1, 5, 0, 2],
    #     [1, 2, 5, 10, 0, 3]
    # ]

    # gap_configs128 = [
    #     [50, 200, 2, 4, 0, 1],
    #     [50, 200, 2, 5, 0, 2],
    #     [1, 2, 5, 15, 0, 3]
    # ]

    # gap_configs64 = [
    #     [50, 200, 1, 4, 0, 1],
    #     [50, 200, 1, 5, 0, 2],
    #     [1, 2, 5, 10, 0, 3]
    # ]

    datagen = image.ImageDataGenerator(
        rescale=1 / 255.,
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    if type == 'DATA_GAP':
        raw_generator_352 = datagen.flow_from_directory(
            './data/line',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            color_mode='grayscale',
            seed=SEED,
            class_mode=None,
            batch_size=BATCH_SIZE,
            shuffle=True,
            interpolation='bilinear'
        )

        # raw_generator_176 = datagen.flow_from_directory(
        #     './data/line',
        #     target_size=(IMG_HEIGHT // 2, IMG_WIDTH // 2),
        #     color_mode='grayscale',
        #     seed=SEED,
        #     class_mode=None,
        #     batch_size=BATCH_SIZE // 2,
        #     shuffle=True,
        #     interpolation='bilinear'
        # )

        while True:
            train_y_batch = next(raw_generator_352)
            train_x_batch, _ = generate_random_gap(train_y_batch, gap_configs352, SEED)

            yield train_x_batch, train_y_batch

    elif type == 'DATA_GEN':
        while True:
            # Size config is in datagen.py
            train_y_batch = gen_data(np.random.RandomState(SEED), BATCH_SIZE)
            train_x_batch, _ = generate_random_gap(train_y_batch, gap_configs352, SEED)

            yield train_x_batch, train_y_batch

    elif type == 'DATA_THIN':
        raw_generator_x = datagen.flow_from_directory(
            './data/thin',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            color_mode='grayscale',
            seed=SEED,
            class_mode=None,
            batch_size=BATCH_SIZE,
            shuffle=True,
            interpolation='bilinear'
        )

        raw_generator_y = datagen.flow_from_directory(
            './data/line',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            color_mode='grayscale',
            seed=SEED,
            class_mode=None,
            batch_size=BATCH_SIZE,
            shuffle=True,
            interpolation='bilinear'
        )

        while True:
            yield next(raw_generator_x), next(raw_generator_y)


def train():
    model = build_model()
    model.summary()

    if LOAD_WEIGHTS:
        model.load_weights('./weight/model1.h5')

    model.compile(loss='MSE', optimizer='Adam')

    data = data_generator(DATA_TYPE)
    start_time = datetime.datetime.now()

    for iteration in range(1, ITERATIONS + 1):

        train_y_batch, train_x_batch = next(data)
        loss = model.train_on_batch(train_x_batch, train_y_batch)

        print('[Time: %s] [Iteration: %d] [Loss: %f]' % (datetime.datetime.now() - start_time, iteration, loss))

        if iteration % 200 == 0:
            model.save('./weight/model1_%d.h5' % iteration)

    model.save('./weight/model1.h5')


if __name__ == "__main__":
    train()
