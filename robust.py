################## How to deal with corrupt files. #####################
########################################################################

import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    df = pd.read_csv('./Data/NoFelicidad.csv')
    print(df.head())

    X = df.drop(['country', 'score'], axis=1)
    y = df['score']
    print('The arrays have the next format', X.shape, y.shape)
    # dt_features= StandardScaler().fit_transform(dt_features)

    # #dt_target= StandardScaler().fit_transform(dt_target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

    #################### Models ############################
    ########################################################

    estimadores={'SVR':SVR(gamma='auto', C=1.0, epsilon=0.1),
               'RANSACR': RANSACRegressor(),
                'HUBBER': HuberRegressor(epsilon=1.35)
               }
    
    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions=estimador.predict(X_test)

        print('='*64)
        print(name)
        print("MSE", mean_squared_error(y_test, predictions))





