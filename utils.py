import pandas as pd
import numpy as np
import sklearn
import joblib
class Utils:

    def load_form_csv(self,path):
        return pd.read_csv(path)
    
    def load_from_mysql():
        pass

    def feature_target(self, data, drop_cols, y):
        X = data.drop(drop_cols, axis=1)
        y = data[y]
        return X, y
    
    def models_exports(self, clf, score):
        print('score')
        joblib.dump(clf,'./models/best_model.pkl')