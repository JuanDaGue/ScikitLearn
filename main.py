from utils import Utils
from models import Model
if __name__ == "__main__":

    utils = Utils()
    models = Model()

    data = utils.load_form_csv('./Data/felicidad.csv')
    print(data)
    X, y = utils.feature_target(data,['score', 'rank', 'country'],['score'] )

    models.grid_trainign(X,y)

