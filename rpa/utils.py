import glob
import imageio
import cv2
import numpy as np
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def format_images(icon_name):
    files = glob.glob("icon_data/%s/*" % icon_name)
    X = np.array(
        [
            preprocess_input(cv2.resize(imageio.imread(filename), (224, 224)))
            for filename in files
        ]
    )
    pickle.dump(X, open("icon_data_resized/%s.p" % icon_name, "wb"))


def load_images(icon_name):
    return pickle.load(open("icon_data_resized/%s.p" % icon_name, "rb"))


def get_icon_names():
    folders = glob.glob("icon_data/*")
    icon_names = []
    for filename in folders:
        icon_names.append(filename[len("icon_data/") :])

    return icon_names


def format_all_classes():

    for icon_name in [get_icon_names()[85]]:
        print("Formatting %s images..." % icon_name)
        format_images(icon_name)


def encode_all_classes():
    icon_names = get_icon_names()
    for icon_name in icon_names:
        print("Encoding %s images..." % icon_name)
        X = load_images(icon_name)
        resnet_model = ResNet50(
            weights="imagenet", include_top=True, input_shape=(224, 224, 3)
        )

        # remove the output layer
        resnet_model.layers.pop()
        model = Model(
            inputs=resnet_model.inputs, outputs=resnet_model.layers[-1].output
        )

        X_encoded = model.predict(X)
        pickle.dump(
            X_encoded, open("icon_data_encoded/%s.p" % icon_name, "wb"),
        )


def load_encoded_images(icon_name):
    return pickle.load(open("icon_data_encoded/%s.p" % icon_name, "rb"))

