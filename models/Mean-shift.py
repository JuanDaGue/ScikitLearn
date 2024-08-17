import pandas as pd

from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("./data/candy.csv")
    print(df.head(5))

    X = df.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)

    print("Max num of labels", max(meanshift.labels_))
    print("="*64)
    print('Center', meanshift.cluster_centers_)
    
    df['meansift'] = meanshift.labels_

    print("="*64)
    print(df)


    #print(meanshift.predict(X))


    pca = PCA(n_components=2)
    pca.fit(X)
    pca_data = pca.transform(X)
        
    meanshift = MeanShift().fit(pca_data)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=meanshift.predict(pca_data))
    plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], c='black', s=200)
    plt.show()