import csv

from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score

import keras 
from keras.layers import LSTM, Embedding, SimpleRNN, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.models import Sequential
import keras.backend as K

var_predictions = ["Polarity", "Attraction"]

names_column_result = ['name_taks', 'model', 'method_da', 'lambda_hyper_1', 'lambda_hyper_2',
       'preprocesamiento', 'representation', 'nrow_df', 'nrow_df_aug','error_train_polarity',
       'error_train_attraction', 'error_train', 'error_test_polarity',
       'error_test_attraction', 'error_test']
param_grid_reg = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

def error_metric_analysis_sentiment(X, y, baselines, name_data):
    errors = {}
    y_predict_pd = {}
    for label in var_predictions:
    	y_predict_pd[label] = baselines[label].predict(X)
    	try:
            y_predict_pd[label] = np.argmax(y_predict_pd[label], axis=1)
        except:
            y_predict_pd[label] = y_predict_pd[label]

        if label=="Polarity":
        	errors["error_"+label] = mean_absolute_error(y_predict[label], y[label])
        else:
        	errors["error_"+label] = f1_score(y_predict[label], y[label], average="macro")

    errors["error_"+name_data]= ((1/(1+errors["error_Polarity"])+errors["error_Attraction"])/2)
    return errors

def selection_representation(X_train=None, X_test=None, representation="TFIDF"):
    if representation=="TFIDF":
        vectorizer = TfidfVectorizer (
          analyzer = 'word',
          min_df = .1,
          max_features = 5000,
          lowercase = True
        )
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
    elif representation=="Tokenizer_Keras":
        max_features = 15000  # number of words to consider as features
        max_len = 500  # cut texts after this number of words (among top max_features most common words)
        #prepara las secuencias
        tokenizer=Tokenizer(nb_words=max_features)
        tokenizer.fit_on_texts(X_train)
        sequencies_tr = tokenizer.texts_to_sequences(X_train)
        sequencies = tokenizer.texts_to_sequences(X_test)

        X_train = sequence.pad_sequences(sequencies_tr, maxlen=max_len)
        X_test = sequence.pad_sequences(sequencies, maxlen=max_len)
    else:
        print("Representaci??n invalida")
        return 0     
    return X_train, X_test

# https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d#:~:text=By%20default%2C%20f1%20score%20is,like%20accuracy%2C%20categorical%20accuracy%20etc.
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def selection_model(X_train, y_train, model_name, number_cate):
    if model_name=="SVM":
        model = SVC(kernel="linear", C=0.01, random_state=19970808)
        model.fit(X_train, y_train)
    elif model_name=="LogisticRegression":
        model = LogisticRegression()
        model = GridSearchCV(model, param_grid = param_grid_reg, cv = 3, verbose=True, n_jobs=-1)
        model.fit(X_train, y_train)
    elif model_name=="CNN":

        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model = Sequential()
        model.add(Embedding(15000, 300, input_length=500))
        model.add(Dropout(0.4))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        if (number_cate-1)==1:
            model.add(Dense(number_cate-1, activation='sigmoid'))
            loss_model = keras.losses.binary_crossentropy
        else:
            model.add(Dense(number_cate, activation='softmax'))
            loss_model = "sparse_categorical_crossentropy"
        model.compile(optimizer='rmsprop',
                      loss=loss_model,
                      metrics=['acc'])
        print(type(X_train))
        model.fit(X_train, np.array(y_train), epochs=2, batch_size=16,
                    validation_split=0.2,
                    callbacks=[callback])
    elif model_name=="RNN":
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model = Sequential()
        model.add(Embedding(15000, 32))
        model.add(LSTM(32))
        model.add(Dense(25,activation='relu'))
        model.add(Dense(number_cate, activation='softmax'))
        loss_model = keras.losses.MeanAbsoluteError()

        model.compile(optimizer='rmsprop',
                      loss=loss_model,
                      metrics=[get_f1])

        model.fit(X_train, np.array(y_train), epochs=2, batch_size=32, validation_split=0.2,
              callbacks=[callback])

    else: 
        print("Modelo inv??lido")

    return model


def train_model_with_accuracy(X_train, X_test, y_train, y_test, model_name):
    # Train model selection
    baselines = {}
    for label in var_predictions:
        number_cate = len(np.unique(y_train[label]))
        # Get a baseline classifier
        baselines[label] = selection_model(X_train, y_train[label], model_name, number_cate) 
    
    # calculate error in train and test
    error_all = error_metric_analysis_sentiment(X_train, y_train[var_predictions], baselines, "train")
    error_test = error_metric_analysis_sentiment(X_test, y_test[var_predictions], baselines, "test")
    for i in error_test:
        error_all.append(i)
    return error_all


def train_model(df=None, col_text=None, model_name=None, method_da=["-1"], lambda_hyper=["-1","-1"], preprocess=["-1"], representation=["TFIDF"], path_errors=None,
    name_path_da=None, name_path_da_preprocesing=None, name_path_preprocesing=None):
    # global vars
    name_task = ["Sentiment_Analysis"]    
    nrow_df = str(df["train"].shape[0])
    # selection method data augmentation
    if method_da[0]!="-1":
        df["train"], nrow_df_aug = data_augment(df["train"], col_text, method_da[0], lambda_hyper)
        X_train = df["train"][col_text]
        df["train"].to_csv(path_save_da+name_path_da, index=False)
    elif method_da[0]=="-1":
        X_train = df["train"][col_text]
        nrow_df_aug = "-1"
    else:
        print("Argumento inv??lido")
    X_test = df["test"][col_text]
    

    # apply preprocessing
    if preprocess[0]!="-1":
        X_train, X_test = apply_preprosesing(X_train, X_test)
        df["train"][col_text] = X_train
        if method_da[0]!="-1":
            df["train"].to_csv(path_save_da_preprocesing+name_path_da_preprocesing, index=False)
        else:
            df["train"].to_csv(path_save_preprocesing+name_path_preprocesing, index=False)

    # representation texts
    X_train, X_test = selection_representation(X_train, X_test, representation[0])
    
    # train and calcule acuracy
    error_all = train_model_with_accuracy(X_train, X_test, df["train"][var_predictions], 
        df["test"][var_predictions], model_name[0])
    
    # format's table    
    result = [name_task, model_name, method_da, lambda_hyper, 
                           preprocess, representation, [nrow_df], [nrow_df_aug], error_all]
    result = [item for sublist in result for item in sublist]
    
    save_result_csv(result, path_errors, names_column_result)
    return 0

def baseline_polities(col_text=None, path_train=None, path_test=None, model_name=None, method_da=["-1"], lambda_hyper=["-1","-1"], preprocess=["-1"], 
    representation=["TFIDF"], path_errors=None):
    set_seed()

    name_path_da = "name_"+method_da[0]+lambda_hyper[0]+lambda_hyper[1]+preprocess[0]+representation[0]+"_da.csv"
    name_path_da_preprocesing = "name_"+method_da[0]+lambda_hyper[0]+lambda_hyper[1]+preprocess[0]+representation[0]+"_da_preprocesing.csv"
    name_path_preprocesing = "name_"+method_da[0]+lambda_hyper[0]+lambda_hyper[1]+preprocess[0]+representation[0]+"_preprocesing.csv"

    paths_da = os.listdir(path_save_da)
    paths_da_preprocesing = os.listdir(path_save_da_preprocesing)
    paths_preprocesing = os.listdir(path_save_preprocesing)
    
    # load file
    train_test = format_data_origin(path_train, path_test)
    nrow_df = str(train_test["train"].shape[0])

    name_task = ["Sentiment_Analysis"]    

    if name_path_da_preprocesing in paths_da_preprocesing: 
        df = pd.read_csv(path_save_da_preprocesing+name_path_da_preprocesing)
        X_train = df[col_text]
        X_test = train_test["test"][col_text]
        X_train, X_test = selection_representation(X_train, X_test, representation[0])

        # train and calcule acuracy
        error_all = train_model_with_accuracy(X_train, X_test, df[var_predictions], 
            train_test["test"][var_predictions], model_name[0])
        
        nrow_df_aug = str(X_train.shape[0])
        # format's table    
        result = [name_task, model_name, method_da, lambda_hyper, 
            preprocess, representation, [nrow_df], [nrow_df_aug], error_all]
        result = [item for sublist in result for item in sublist]
        
        save_result_csv(result, path_errors, names_column_result)
 
    elif name_path_da in paths_da:
        df = pd.read_csv(path_save_da+name_path_da)
        X_train  = df[col_text]
        X_test = train_test["test"][col_text]

        if preprocess[0]!="-1":
            X_train, X_test = apply_preprosesing(X_train, X_test)
            df[col_text] = X_train
            df.to_csv(path_save_da_preprocesing+name_path_da_preprocesing, index=False)
        # representation texts
        X_train, X_test = selection_representation(X_train, X_test, representation[0])
        
        # train and calcule acuracy
        error_all = train_model_with_accuracy(X_train, X_test, df[var_predictions], 
             train_test["test"][var_predictions], model_name[0])
        
        # format's table   
        nrow_df_aug = str(X_train.shape[0])
 
        result = [name_task, model_name, method_da, lambda_hyper, 
                               preprocess, representation, [nrow_df], [nrow_df_aug], error_all]
        result = [item for sublist in result for item in sublist]
        
        save_result_csv(result, path_errors, names_column_result)

    elif name_path_preprocesing in paths_preprocesing:
        df = pd.read_csv(path_save_preprocesing+name_path_preprocesing)
        X_train = df[col_text]
        X_test = train_test["test"][col_text]
        X_train, X_test = selection_representation(X_train, X_test, representation[0])

        # train and calcule acuracy
        error_all = train_model_with_accuracy(X_train, X_test, df[var_predictions], 
            train_test["test"][var_predictions], model_name[0])
        
        # format's table
        nrow_df_aug = str(X_train.shape[0])
        result = [name_task, model_name, method_da, lambda_hyper, 
            preprocess, representation, [nrow_df], [nrow_df_aug], error_all]
        result = [item for sublist in result for item in sublist]
        
        save_result_csv(result, path_errors, names_column_result)
 
    else: 
        train_model(train_test, col_text, model_name, method_da, lambda_hyper, preprocess, 
    representation, path_errors, name_path_da, name_path_da_preprocesing, name_path_preprocesing)
    
    return 0
