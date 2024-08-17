################## Bagging Method. #####################
########################################################################

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    df = pd.read_csv('E:/DataScience/ScikitLearn/Data/heart.csv')
    print(df['target'].describe())
    X = df.drop(['target'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =50)

    # K means

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)

    print('='*36)
    print(accuracy_score(knn_pred, y_test))

    # Bagging class

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators =50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)

    print('='*36)
    print(accuracy_score(bag_pred, y_test))
