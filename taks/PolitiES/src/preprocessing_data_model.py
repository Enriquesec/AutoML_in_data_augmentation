import pandas as pd 
from my_functions import * 
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# var_global_general
path_save_da = "../data/processed/da/"
path_save_da_preprocesing = "../data/processed/da_preprocesing/"
path_save_preprocesing = "../data/processed/preprocesing/"

dic_gender = {"male":0, "female":1}
dic_profession = {"journalist":0, "politician":1}
dic_ideology_binary = {"right":0, "left":1}
dic_idiology_multiclas = {'moderate_right':0, 'moderate_left':1, 
                          'right':2, 'left':3}

dic_complete = [dic_gender, dic_profession, dic_ideology_binary, dic_idiology_multiclas]
var_predictions = ['gender', 'profession', 'ideology_binary', 'ideology_multiclass']

def category_col(df=None, var_predictions=['gender', 'profession', 'ideology_binary', 'ideology_multiclass']):
  for i, name in enumerate(var_predictions):
    df[name] = df[name].apply(lambda x: dic_complete[i][x])
  'gender', 'profession', 'ideology_binary', 'ideology_multiclass'
  return df 

def format_data_origin(path_train, path_test):
    reviews_politica = pd.read_csv(path_train)
    reviews_politica_test = pd.read_csv(path_test)

    # From now one, dataframes will contain the dataframes
    dataframes = {
      'train': reviews_politica, 
      'test': reviews_politica_test
    }

    # NOTE: As loops does not bind variable data, we do sequence unpacking
    for key, df in dataframes.items():

      # These columns are shared for all documents of each user
      columns_to_group_by_user = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']


      # Group the dataframe by user (label)
      group = df.groupby (by = columns_to_group_by_user, dropna = False, observed = True, sort = False)


      # Create a custom dataframe per user
      df_users = group[columns_to_group_by_user].agg (func = ['count'], as_index = False, observed = True).index.to_frame (index = False)


      # Temporal variable
      merged_fields = []


      # We merge the documents with a fancy TQDM progress bar
      # pbar = tqdm (df_users.shape[0], total = df_users.shape[0], desc = "merging users")


      # Iterate over rows in a fancy w
      for index, row in df_users.iterrows():
          df_user = df[(df['label'] == row['label'])]
          merged_fields.append ({**row, **{field: ' [SEP] '.join (df_user[field].fillna ('')) for field in ['tweet']}})

      # Modify the original variable dataframe
      dataframes[key] = category_col(pd.DataFrame(merged_fields))
    return dataframes


def apply_preprosesing(X_train,X_test):
    preprocesador = preprocesaTexto(idioma='es', _tokeniza=False, _muestraCambios=False, _quitarAcentos=True,
                                _quitarNumeros=False, _remueveStop=True, _stemming=True, _lematiza=True,
                                _removerPuntuacion=True)
    X_train = [preprocesador.preprocesa(i) for i in X_train] 
    X_test = [preprocesador.preprocesa(i) for i in X_test]
    return X_train, X_test