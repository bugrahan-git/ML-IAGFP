import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from Util import get_images
from plot import line
import Util


# Preparing the Initial Data
# Util.prepare_data(isAugmented=True)

# Feature Extraction
# X_train, y_train, X_test, y_test = Util.extract_features_artist()
# X_train, y_train, X_test, y_test = Util.extract_features_movement()


X_train, y_train = get_images("train_artist.mat", "artists")
X_test, y_test = get_images("test_artist.mat", "artists")

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

dict_artist_naive_bayes = {
    "jackson-pollock": [0, 0],
    "wassily-kandinsky": [0, 0],
    "fernand-leger": [0, 0],
    "pablo-picasso": [0, 0],
    "claude-monet": [0, 0],
    "pierre-auguste-renoir": [0, 0],
    "jacques-louis-david": [0, 0],
    "jean-auguste-dominique-ingres": [0, 0],
    "andy-warhol": [0, 0],
    "roy-lichtenstein": [0, 0],
    "mikhail-vrubel": [0, 0],
    "odilon-redon": [0, 0],
}

for i in range(len(y_pred)):
    dict_artist_naive_bayes[y_test[i].strip()][0] += 1
    if y_pred[i] == y_test[i]:
        dict_artist_naive_bayes[y_test[i].strip()][1] += 1

for key, val in dict_artist_naive_bayes.items():
    print(key, " ", val)

print("ARTIST CLASSIFICATION NAIVE_BAYES")
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
score = accuracy_score(y_test, y_pred)
print("Score = ", score)

print("Weighted Knn")
k_range = range(1, 15)
scores = {}
score_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    score_list.append(metrics.accuracy_score(y_test, y_pred))


print(score_list)
line(k_range, score_list, "K", "Accuracy", color="yellowgreen", title="Weighted K_nn Accuracy Graph")


print("Knn")
k_range = range(1, 15)
scores = {}
score_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    score_list.append(metrics.accuracy_score(y_test, y_pred))


print(score_list)
line(k_range, score_list, "K", "Accuracy", color="yellowgreen", title="K_nn Accuracy Graph")

"""

"""
print("MOVEMENT CLASSIFICATION")
X_train, y_train = get_images("train_movement.mat", "movements")
X_test, y_test = get_images("test_movement.mat", "movements")

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


dict_movement_naive_bayes = {
    "abstractart": [0, 0],
    "cubism": [0, 0],
    "impressionism": [0, 0],
    "neoclassicism": [0, 0],
    "popart": [0, 0],
    "symbolism": [0, 0],
}

for i in range(len(y_pred)):
    dict_movement_naive_bayes[y_test[i].strip()][0] += 1
    if y_pred[i] == y_test[i]:
        dict_movement_naive_bayes[y_test[i].strip()][1] += 1

for key, val in dict_movement_naive_bayes.items():
    print(key, " ", val)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
score = accuracy_score(y_test, y_pred)
print("Score = ", score)



print("Weighted Knn")
k_range = range(1, 15)
scores = {}
score_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    score_list.append(metrics.accuracy_score(y_test, y_pred))


print(score_list)
line(k_range, score_list, "K", "Accuracy", color="red", title="Weighted K_nn Accuracy Graph")


print("Knn")
k_range = range(1, 15)
scores = {}
score_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    score_list.append(metrics.accuracy_score(y_test, y_pred))


print(score_list)
line(k_range, score_list, "K", "Accuracy", color="red", title="K_nn Accuracy Graph")
