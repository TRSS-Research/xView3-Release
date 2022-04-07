import tensorflow as tf
from tensorflow.keras import  applications
from src.classifier.data_loader import XviewSequence, XviewSequenceInMemory
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

    train_data = XviewSequence(f'{directory}/training', channels_ix=(static.VV, static.VH), batch_size=batch_size,
                               as_bool=False, image_augmenter=train_datagen)
    valid_data = XviewSequence(f'{directory}/validation', channels_ix=(static.VV, static.VH), batch_size=batch_size,
                               as_bool=False, image_augmenter=valid_datagen, train=False)

# alternatively, if you have a very large amount of RAM:
    if args.in_memory:
        train_data = XviewSequenceInMemory(f'{directory}/training', channels_ix=(static.VV, static.VH),
                                           batch_size=batch_size, as_bool=False, image_augmenter=train_datagen)
        valid_data = XviewSequenceInMemory(f'{directory}/validation', channels_ix=(static.VV, static.VH),
                                           batch_size=batch_size, as_bool=False, image_augmenter=valid_datagen,
                                           train=False)

    # Define model
    if load_path is None:
        vggnet = applications.vgg16.VGG16(input_shape=train_data.input_data_shape,
                                          include_top=True,
                                          weights=None,
                                          classes=4)
    else:
        vggnet = load_model(load_path)

    vggnet.compile(optimizer='sgd',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])

    print('Model Compiled')
    vggnet.summary()

    vgg16_history = vggnet.fit(x=train_data, epochs=epochs, verbose=1, validation_data=valid_data)
    print("Training Complete!")
    # Retrieve a list of accuracy results on training and validation data
    # sets for each training epoch

    os.makedirs(f'{save_path}/metrics', exist_ok=True)
    os.makedirs(f'{save_path}/model', exist_ok=True)
    
    acc = vgg16_history.history['accuracy']  # blue
    val_acc = vgg16_history.history['val_accuracy']  # orange

    plt.plot(list(range(epochs)), acc, label='Training')
    plt.plot(list(range(epochs)), val_acc, label='Validation')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/metrics/accuracy_at_{epochs}.png')

    loss = vgg16_history.history['loss']  # blue
    val_loss = vgg16_history.history['val_loss']  # orange

    plt.plot(list(range(epochs)), loss, label='Training')
    plt.plot(list(range(epochs)), val_loss, label='Validation')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'{save_path}/metrics/loss_at_{epochs}.png')

    print('saving')
    vggnet.save(f'{save_path}/SAR-VGG16.model')
    model_json = vggnet.to_json()
    with open(f"{save_path}/model/SAR-VGG16.json", "w") as json_file:
        json_file.write(model_json)
    vggnet.save_weights(f'{save_path}/model/SAR-VGG16.weights')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--directory',default='data', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--save_path', default='output/models/CFAR-VGG16', type=str)
    parser.add_argument('--prebuilt_model_path', default=None, type=str)
    parser.add_argument('--in_memory', default=False, type=bool)
    args = parser.parse_args()
    print(args)
    main(args)

