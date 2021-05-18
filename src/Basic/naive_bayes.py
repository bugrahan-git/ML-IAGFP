from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def artist_identification(X_train, y_train, X_test, y_test):
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


def movement_identification(X_train, y_train, X_test, y_test):
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