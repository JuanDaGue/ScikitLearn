import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    df = pd.read_csv('./Data/felicidad.csv')
    print(df.head())

    X =df[['gdp','family','lifexp','freedom','corruption','dystopia']]
    y =df['score']
    print('The arrays have the next format',X.shape,y.shape)
    # dt_features= StandardScaler().fit_transform(dt_features)

    # #dt_target= StandardScaler().fit_transform(dt_target)

    X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=0.25,random_state=50)

    #################### Mdels ############################
    ########################################################


    Model_lineal=LinearRegression().fit(X_train,y_train)
    y_predict= Model_lineal.predict(X_test)
    
    
    model_lasso= Lasso(alpha=0.032).fit(X_train,y_train)
    y_predictLasso= model_lasso.predict(X_test)

    model_ridge= Ridge(alpha=1).fit(X_train,y_train)
    y_predictridge= model_ridge.predict(X_test)

    linear_lose = mean_squared_error(y_test, y_predict)

    print('Linear lose', linear_lose)

    linear_loseLasso = mean_squared_error(y_test, y_predictLasso)

    print('Lasso lose', linear_loseLasso)

    linear_loseRidge = mean_squared_error(y_test, y_predictridge)

    print('Ridge lose', linear_loseRidge)
    print('='*32)
    print('Coef Lasso')
    print(model_lasso.coef_) 
    print('='*32)   
    print('Coef Ridge')
    print(model_ridge.coef_)