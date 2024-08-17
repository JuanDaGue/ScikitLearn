import pandas as pd
import sklearn

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    df = pd.read_csv('./Data/heart.csv')
    print(df['target'].describe())
    X = df.drop(['target'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =50)

    # Boosting

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)

    
    print('='*36)
    print(accuracy_score(boost_pred, y_test))