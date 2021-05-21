from sklearn.preprocessing import LabelBinarizer
from src.Util.Util import get_images, prepare_data, extract_features_artist, extract_features_movement
from Basic import naive_bayes, knn
from Model.lenet5 import LeNet5
from Model.alexnet import Alexnet
from Util import plot
import csv


def prepare(isAugmented=True):
    print(f"Preparing the data, Augmenting={not isAugmented}")
    prepare_data(isAugmented=isAugmented)


def getImages(extract_features=False, isArtist=True):
    if isArtist:
        if extract_features:
            X_train, y_train, X_validation, y_validation, X_test, y_test = extract_features_artist()
        X_train, y_train = get_images("../mat/train_artist.mat", "artists")
        X_test, y_test = get_images("../mat/test_artist.mat", "artists")
        X_validation, y_validation = get_images("../mat/validation_artist.mat", "artists")
    else:
        if extract_features:
            X_train, y_train, X_validation, y_validation, X_test, y_test = extract_features_movement()
        X_train, y_train = get_images("../mat/train_movement.mat", "movements")
        X_test, y_test = get_images("../mat/test_movement.mat", "movements")
        X_validation, y_validation = get_images("../mat/validation_movement.mat", "movements")

    return X_train, y_train, X_validation, y_validation, X_test, y_test

"""
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
"""

"""
# get movement dataset
X_train_movement, y_train_movement, X_validation_movement, y_validation_movement, X_test_movement, y_test_movement = getImages(
    extract_features=False, isArtist=False)
"""

encoder = LabelBinarizer()


# prepare()

def run(EPOCHS, BATCH_SIZE, ACTIVATION_FUNCTION, MODEL):
    X_train_artist, y_train_artist, X_validation_artist, y_validation_artist, X_test_artist, y_test_artist = getImages(
        extract_features=False, isArtist=True)

    # Artist identification
    X_train, y_train = X_train_artist.reshape(X_train_artist.shape[0], 32, 32, 1), encoder.fit_transform(y_train_artist)
    X_validation, y_validation = X_validation_artist.reshape(X_validation_artist.shape[0], 32, 32,
                                                             1), encoder.fit_transform(y_validation_artist)
    X_test, y_test = X_test_artist.reshape(X_test_artist.shape[0], 32, 32, 1), encoder.fit_transform(y_test_artist)


    if MODEL == "LeNet-5": # Create LeNet model
        model = LeNet5(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION)


    if MODEL == "AlexNet": # create AlexNet model
        model = AlexNet(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, CLASS_COUNT=12)

    model.summary()

    # Train the model

    training_loss, validation_loss, training_accuracy, validation_accuracy = model.train(X_train, y_train, X_validation,
                                                                                         y_validation)
    
    plot.line(training_loss, validation_loss, "Epochs", "Loss",
              f"Model={MODEL} Artist Loss Graph\nEpoch={EPOCHS},Batch Size={BATCH_SIZE},Activation Function={ACTIVATION_FUNCTION}")
    plot.line(training_accuracy, validation_accuracy, "Epochs", "Accuracy",
              f"Model={MODEL} Artist Accuracy Graph\nEpoch={EPOCHS},Batch Size={BATCH_SIZE},Activation Function={ACTIVATION_FUNCTION}")

    with open('../train_results.csv', 'a', newline='') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
        csvWriter.writerow(
            [MODEL, str(EPOCHS), str(BATCH_SIZE), ACTIVATION_FUNCTION, " ".join(str(x) for x in training_loss),
             " ".join(str(x) for x in validation_loss),
             " ".join(str(x) for x in training_accuracy),
             " ".join(str(x) for x in validation_accuracy)])


with open('../test_lenet5.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    ctr = False
    for row in csv_reader:
        if not ctr:
            ctr = True
        else:
            # run(EPOCHS=int(row[1]), BATCH_SIZE=int(row[0]), ACTIVATION_FUNCTION=row[2], MODEL="LeNet-5")
            run(EPOCHS=int(row[1]), BATCH_SIZE=int(row[0]), ACTIVATION_FUNCTION=row[2], MODEL="AlexNet")


