import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('E:/DataScience/ScikitLearn/Data/heart.csv')
    print(dt_heart.head())

    dt_features =dt_heart.drop(['target'],axis=1)
    dt_target =dt_heart['target']

    dt_features= StandardScaler().fit_transform(dt_features)

    #dt_target= StandardScaler().fit_transform(dt_target)

    X_train,X_test, y_train, y_test= train_test_split(dt_features,dt_target,test_size=0.3,random_state=50)
    pca = PCA(n_components=3)
    pca.fit(X_train)
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_)
    #plt.show()

    logistic=LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train,y_train)

    print("Score PCA" , logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train,y_train)

    print("Score IPCA" , logistic.score(dt_test, y_test))