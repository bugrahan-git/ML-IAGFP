from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from src.Util.plot import line


def knn(X_train, y_train, X_test, y_test, weights="distance"):
    k_range = range(1, 15)
    scores = {}
    score_list = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        score_list.append(metrics.accuracy_score(y_test, y_pred))

    print(score_list)
    line(k_range, score_list, "K", "Accuracy", color="yellowgreen", title="Accuracy Graph")
