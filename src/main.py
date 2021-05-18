from src.Util.Util import get_images, prepare_data, extract_features_artist, extract_features_movement
from Basic import naive_bayes, knn


def prepare(isAugmented=True):
    print(f"Preparing the data, Augmenting={isAugmented}")
    prepare_data(isAugmented=True)


def getImages(extract_features=False, isArtist=True):
    if isArtist:
        if extract_features:
            X_train, y_train, X_test, y_test = extract_features_artist()
        X_train, y_train = get_images("dataset/mat/train_artist.mat", "artists")
        X_test, y_test = get_images("dataset/mat/test_artist.mat", "artists")
    else:
        if extract_features:
            X_train, y_train, X_test, y_test = extract_features_movement()
        X_train, y_train = get_images("dataset/mat/train_movement.mat", "movements")
        X_test, y_test = get_images("dataset/mat/test_movement.mat", "movements")

    return X_train, y_train, X_test, y_test


def artist_identification_with_basic_algorithms():
    # naive bayes classification for artists
    naive_bayes.artist_identification(X_train_artist, y_train_artist, X_test_artist, y_test_artist)
    # weighted knn classification for artists
    knn.knn(X_train_artist, y_train_artist, X_test_artist, y_test_artist)
    # knn classification for artists
    knn.knn(X_train_artist, y_train_artist, X_test_artist, y_test_artist, weights="uniform")


def movement_identification_with_basic_algorithms():
    # naive bayes classification for movements
    naive_bayes.movement_identification(X_train_movement, y_train_movement, X_test_movement, y_test_movement)
    # weighted knn classification for movements
    knn.knn(X_train_movement, y_train_movement, X_test_movement, y_test_movement)
    # knn classification for movements
    knn.knn(X_train_movement, y_train_movement, X_test_movement, y_test_movement, weights="uniform")


# get artist dataset
X_train_artist, y_train_artist, X_test_artist, y_test_artist = getImages()
# get movement dataset
X_train_movement, y_train_movement, X_test_movement, y_test_movement = getImages(extract_features=False, isArtist=False)
