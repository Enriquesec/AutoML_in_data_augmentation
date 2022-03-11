import pandas as pd 
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import os
os.chdir(r"/home/enrique/Documentos/AutoML_in_data_aumentation/taks/Agresivite/src/")

from test_evaluation import *
from my_functions import * 

var_predictions = ["label"]


def read_data_origin(path_train, path_test):
    reviews_politica = pd.read_csv(path_train)
    reviews_politica_test = pd.read_csv(path_test)

    # From now one, dataframes will contain the dataframes
    dataframes = {
      'train': reviews_politica, 
      'test': reviews_politica_test
    }
   
    return dataframes

def noise_injection(lambda_hyper):
    aug = naw.RandomWordAug(action=lambda_hyper[0], aug_p=lambda_hyper[1])
    return aug

def replacing_synonym(lambda_hyper):
    aug = naw.SynonymAug(aug_src='wordnet',  lang="spa", aug_p=lambda_hyper[1])
    return aug

def back_translation(lambda_hyper):
    aug = naw.BackTranslationAug(
    from_model_name=lambda_hyper[0], 
    to_model_name=lambda_hyper[1])
    return aug

def selecction_method_aug(taks_augment, lambda_hyper):
    if taks_augment=="noise_injection":
        aug = noise_injection(lambda_hyper)
    elif taks_augment=="replacing_synonym":
        aug = replacing_synonym(lambda_hyper)
    elif taks_augment=="back_translation":
        aug = back_translation(lambda_hyper)
    else:
        print("Error")
    return aug

def data_augment(df, col_int, taks_augment, lambda_hyper):
    aug = selecction_method_aug(taks_augment, lambda_hyper)
    df_augment = []
    for i in range(df.shape[0]):
        df_augment.append(aug.augment(df.loc[i,col_int][0:500], n=1, num_thread=3))
    
    df_aug = df.copy()
    df_aug[col_int] = df_augment 
    
    df_cont = pd.concat([df, df_aug])
    
    return df_cont.drop_duplicates()


def error_train_test(df=None, method_da=["-1"], lambda_hyper=["-1","-1"], preprocess=["-1"], representation=["TFIDF"], path_errors=None):
    # global vars
    name_task = ["Agresivite"]
    model=["SVC"]
    representation=["TFIDF"]
    
    if method_da[0]!="-1":
        df["train"] = data_augment(df["train"],"tweet", method_da[0], lambda_hyper)
        X_train = df["train"]["tweet"]
    elif method_da[0]=="-1":
        X_train = df["train"]["tweet"]
    else:
        print("Argumento inválido")
    X_test = df["test"]["tweet"]
    
    if preprocess[0]!="-1":
        preprocesador = preprocesaTexto(idioma='es', _tokeniza=False, _muestraCambios=False, _quitarAcentos=True,
                                _quitarNumeros=False, _remueveStop=True, _stemming=True, _lematiza=True,
                                _removerPuntuacion=True)
        # Get the TF-IDF values from the training set
        X_train = [preprocesador.preprocesa(i) for i in X_train] 

        # Get the TF-IDF values from the test set
        # Note that we apply the TF-IDF learned from the training split 
        X_test = [preprocesador.preprocesa(i) for i in X_test]
    
    # Create a TFIDF Vectorizer using sci-kit. With this, we are going to represent all texts
    # as counts of the vocabulary. 
    vectorizer = TfidfVectorizer (
      analyzer = 'word',
      min_df = .1,
      max_features = 5000,
      lowercase = True
    )
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    # We are going to store a baseline per trait
    baselines = {}
    # As we observed, this task is about four traits: two demographic and two psychographic. Therefore, we are going to
    # train different and separate models for each task

    for label in var_predictions:
        # Get a baseline classifier
        baselines[label] = SVC() # LogisticRegression() 
        # Train the baseline for this label
        baselines[label].fit(X_train, df["train"][label])
    
    # calculate error in train and test
    error_all = calculate_error(X_train, df["train"][var_predictions], baselines, "train")
    error_test = calculate_error(X_test, df["test"][var_predictions], baselines, "train")
    
    for i in error_test:
        error_all.append(i)
        
    result = [name_task, model, method_da, lambda_hyper, 
                           preprocess, representation, error_all]
    result = [item for sublist in result for item in sublist]
    

    file = pd.read_csv(path_errors)

    file.loc[file.shape[0]] = result
    file.to_csv(path_errors, index=False)
    return 0
