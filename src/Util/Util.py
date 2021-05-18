import os
from collections import Counter
from random import random
import cv2
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from src.Util.GIST import GIST
from src.Util.Transform import Transform

bins = 8
param = {
    "orientationsPerScale": np.array([8, 8]),
    "numberBlocks": [10, 10],
    "fc_prefilt": 10,
    "boundaryExtension": 32
}

fixed_size = tuple((250, 250))

image_list = list()
artist_list = list()
movement_list = list()
feature_list = list()


def get_movement(artist):
    helper = {
        "abstractart": ["jackson-pollock", "wassily-kandinsky"],
        "cubism": ["fernand-leger", "pablo-picasso"],
        "impressionism": ["claude-monet", "pierre-auguste-renoir"],
        "neoclassicism": ["jacques-louis-david", "jean-auguste-dominique-ingres"],
        "popart": ["andy-warhol", "roy-lichtenstein"],
        "symbolism": ["mikhail-vrubel", "odilon-redon"]
    }

    for key, val in helper.items():
        if artist in val:
            return key


def get_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def get_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


def get_gist(input_image):
    image = np.array(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY))
    gist = GIST(param)
    gist_feature = gist._gist_extract(image)
    return np.array(gist_feature)


def prepare_data(isAugmented=False):
    artists_dict = {}
    movements_dict = {}

    print("Preparing the data...")
    movements = os.listdir("../../dataset")
    for movement in movements:
        artists = os.listdir("dataset/" + movement)
        movement_tmp = []
        for artist in artists:
            images = os.listdir("dataset/" + movement + "/" + artist)
            artist_tmp = []
            for image in images:
                img = cv2.imread("dataset/" + movement + "/" + artist + "/" + image)
                if img is not None:
                    img = cv2.resize(img, fixed_size)
                    image_list.append(img)
                    artist_tmp.append(img)
                    artist_list.append(artist)
                    movement_list.append(movement)
                    movement_tmp.append(img)
                else:
                    print(artist, " ", image, " None")
            artists_dict[artist] = artist_tmp
        movements_dict[movement] = movement_tmp

    if not isAugmented:
        print("Augmenting the data...")
        t = Transform()
        counter = Counter(artist_list)
        max_key = max(counter, key=counter.get)
        for artist in counter.keys():
            artist_movement = get_movement(artist)
            if counter[artist] < counter[max_key]:
                for i in range(counter[max_key] - counter[artist]):
                    transformed_image = t.transform_image(random.choice(artists_dict[artist]),
                                                          "dataset/" + artist_movement + "/" + artist)
                    image_list.append(transformed_image)
                    movement_list.append(artist_movement)
                    artist_list.append(artist)

    X_train_artist, X_test_artist, y_train_artist, y_test_artist = train_test_split(image_list, artist_list,
                                                                                    test_size=0.2, random_state=42)

    c = list(zip(X_train_artist, y_train_artist))
    train_images_artist, train_labels_artist = zip(*c)

    train_artist = {"images": train_images_artist, "artists": train_labels_artist}
    io.savemat(f"dataset/mat/train_artist.mat", train_artist)

    c = list(zip(X_test_artist, y_test_artist))
    test_images_artist, test_labels_artist = zip(*c)

    test_artist = {"images": test_images_artist, "artists": test_labels_artist}
    io.savemat(f"dataset/mat/test_artist.mat", test_artist)

    X_train_movement, X_test_movement, y_train_movement, y_test_movement = train_test_split(image_list, movement_list,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

    c = list(zip(X_train_movement, y_train_movement))
    train_images_movement, train_labels_movement = zip(*c)

    train_movement = {"images": train_images_movement, "movements": train_labels_movement}
    io.savemat(f"dataset/mat/train_movement.mat", train_movement)

    c = list(zip(X_test_movement, y_test_movement))
    test_images_movement, test_labels_movement = zip(*c)

    test_movement = {"images": test_images_movement, "movements": test_labels_movement}
    io.savemat(f"dataset/mat/test_movement.mat", test_movement)


def get_feature_vec(arr):
    hu_moments_list = []
    histogram_list = []
    gist_list = []
    counter = 0
    for img in arr:
        hu_moments_list.append(get_hu_moments(img))
        histogram_list.append(get_histogram(img))
        gist_list.append(get_gist(img))
        counter += 1
        if counter % 100 == 0:
            print(counter)

    feature_vec = np.hstack([np.vstack(hu_moments_list), np.vstack(histogram_list), np.vstack(gist_list)])
    return feature_vec


# get images from .mat file
def get_images(filepath, artistOrMovement):
    images = io.loadmat(filepath)
    image_list = np.array(images["images"])
    label_list = np.array(images[artistOrMovement])
    return image_list, label_list


def extract_features_artist():
    X_train, y_train = get_images("dataset/mat/train_artist.mat", "artists")
    X_train = get_feature_vec(X_train)

    c = list(zip(X_train, y_train))
    train_images_artist, train_labels_artist = zip(*c)

    train_artist = {"images": train_images_artist, "artists": train_labels_artist}
    io.savemat(f"dataset/mat/train_artist.mat", train_artist)

    X_test, y_test = get_images("dataset/mat/test_artist.mat", "artists")
    X_test = get_feature_vec(X_test)

    c = list(zip(X_test, y_test))
    test_images_artist, test_labels_artist = zip(*c)

    test_artist = {"images": test_images_artist, "artists": test_labels_artist}
    io.savemat(f"dataset/mat/test_artist.mat", test_artist)

    return X_train, y_train, X_test, y_test


def extract_features_movement():
    X_train, y_train = get_images("dataset/mat/train_movement.mat", "movements")
    X_train = get_feature_vec(X_train)

    c = list(zip(X_train, y_train))
    train_images_movement, train_labels_artist = zip(*c)

    train_artist = {"images": train_images_movement, "movements": train_labels_artist}
    io.savemat(f"dataset/mat/train_movement.mat", train_artist)

    X_test, y_test = get_images("dataset/mat/test_movement.mat", "movements")
    X_test = get_feature_vec(X_test)

    c = list(zip(X_test, y_test))
    test_images_movement, test_labels_movement = zip(*c)

    test_artist = {"images": test_images_movement, "movements": test_labels_movement}
    io.savemat(f"dataset/mat/test_movement.mat", test_artist)

    return X_train, y_train, X_test, y_test
