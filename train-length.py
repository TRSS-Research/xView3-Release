import tensorflow as tf
from keras import models, layers
from src.classifier.data_loader import XviewLength
from src.classifier.nn_classifier import load_model
from keras.preprocessing.image import ImageDataGenerator
from argparse import ArgumentParser
from src import static
import os

import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

# Only transformations applied to validation is rescaling!
valid_datagen = ImageDataGenerator()


def main(args):
    directory = args.directory
    batch_size = args.batch_size
    epochs = args.epochs
    save_path = args.save_path
    load_path = args.prebuilt_model_path
    train_data = XviewLength(f'{directory}/training', channels_ix=(static.VV, static.VH, static.BATHYMETRY),
                             batch_size=batch_size, as_bool=False, image_augmenter=train_datagen)
    valid_data = XviewLength(f'{directory}/validation', channels_ix=(static.VV, static.VH, static.BATHYMETRY),
                             batch_size=batch_size, as_bool=False, image_augmenter=valid_datagen)


    # Feel free to add metrics if you'd like to monitor other parts of the learning



    # Define model
    if load_path is None:
        model_length = models.Sequential([
            layers.Conv2D(16, kernel_size=2, strides=2, padding='same', activation='relu',
                          input_shape=train_data.input_data_shape),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(0.2),

            layers.Conv2D(32, kernel_size=2, strides=2, activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(0.2),

            layers.Conv2D(32, kernel_size=2, strides=2, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(0.2),

            layers.Flatten(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear'),
        ])

    else:
        model_length = load_model(load_path)

    model_length.compile(loss=tf.keras.losses.mean_squared_error,
                         optimizer=tf.keras.optimizers.Adam(),
                         metrics=['mse'])
    model_length.summary()





    mhistory_length  = model_length .fit(x=train_data, epochs=epochs, verbose=1, validation_data=valid_data)
    print("Training Complete!")
    # Retrieve a list of accuracy results on training and validation data
    # sets for each training epoch

    os.makedirs(f'{save_path}/metrics', exist_ok=True)
    os.makedirs(f'{save_path}/model', exist_ok=True)

    mse = mhistory_length.history['mse']  # blue
    val_mse = mhistory_length.history['val_mse']  # orange

    plt.plot(list(range(epochs)), mse, label='Training')
    plt.plot(list(range(epochs)), val_mse, label='Validation')
    plt.title('Training and validation mse')
    plt.legend()
    plt.savefig(f'{save_path}/metrics/mse_at_{epochs}.png')

    loss = mhistory_length.history['loss']  # blue
    val_loss = mhistory_length.history['val_loss']  # orange

    plt.plot(list(range(epochs)), loss, label='Training')
    plt.plot(list(range(epochs)), val_loss, label='Validation')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'{save_path}/metrics/loss_at_{epochs}.png')

    print('saving')
    model_length.save(f'{save_path}/SAR-length.model')
    model_json = model_length.to_json()
    with open(f"{save_path}/model/SAR-length.json", "w") as json_file:
        json_file.write(model_json)
    model_length.save_weights(f'{save_path}/model/SAR-length.weights')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--directory', default='data', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--save_path', default='output/models/CFAR-length', type=str)
    parser.add_argument('--prebuilt_model_path', default=None, type=str)
    args = parser.parse_args()
    print(args)
    main(args)

