import pandas as pd 
from my_functions import * 
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# var_global_general
path_save_da = "../data/processed/da/"
path_save_da_preprocesing = "../data/processed/da_preprocesing/"
path_save_preprocesing = "../data/processed/preprocesing/"

dic_attraction = {'Hotel':0, 'Restaurant':1, 'Attractive':2}
var_predictions = ["Polarity", "Attraction"]

def format_data_origin(path_train, col_text):
    reviews_politica = pd.read_csv(path_train)
    X_train, X_test, y_train, y_test = train_test_split(reviews_politica[col_text], reviews_politica[var_predictions], 
      test_size=0.20, random_state=19970808)

    # From now one, dataframes will contain the dataframes
    dataframes = {
      'train': pd.concat([X_train, y_train], axis=1), 
      'test': pd.concat([X_test, y_test], axis=1)
    }

    return dataframes


def apply_preprosesing(X):
    preprocesador = preprocesaTexto(idioma='es', _tokeniza=False, _muestraCambios=False, _quitarAcentos=True,
                                _quitarNumeros=False, _remueveStop=True, _stemming=True, _lematiza=True,
                                _removerPuntuacion=True)
    X_pre = [preprocesador.preprocesa(i) for i in X] 
    return X_pre