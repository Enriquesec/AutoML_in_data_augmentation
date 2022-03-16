import random
import numpy as np 
import csv
import pandas as pd

# diccionarios
model_sel = ['SVM', 'LogisticRegression', 'CNN', 'RNN']
method_da = ['noise_injection', 'replacing_synonym', 'back_translation', 'replacing_embeddings', '-1']
lambda_hyper = {'noise_injection': [['delete', '0.3'], ['delete', '0.3']], 
'replacing_synonym': [['-1', '0.3']],
'-1': [['-1', '-1']],
'back_translation': [['Helsinki-NLP/opus-mt-es-en','Helsinki-NLP/opus-mt-en-es']],
 'replacing_embeddings': [['fasttext', '-1'], ['glove', '-1'], ['word2vect', '-1']]}
preprocesamiento = ['1', '-1']
representation = {'SVM': ['TFIDF', 'Tokenizer_Keras'], 
'LogisticRegression': ['TFIDF', 'Tokenizer_Keras'],
'CNN': ['Tokenizer_Keras'],
'RNN': ['Tokenizer_Keras']}


def set_seed(seed=19970808):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)


def save_result_csv(result, path_errors, names_column_result):
    # validation file results.
    try:
        file = pd.read_csv(path_errors)
    except:
        with open(path_errors, 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            header = names_column_result
            file_writer.writerow(header)
        file = pd.read_csv(path_errors)

    file.loc[file.shape[0]] = result
    file.to_csv(path_errors, index=False)

def create_taks_sbatch(path_train_a, path_test_a, path_error_a, path_task):
    model_result = pd.read_csv(path_error_a)
    f = open(path_task, "w")
    for model_name in model_sel:
        for da in method_da:
            for da_sel in lambda_hyper[da]:
                for pre in preprocesamiento:
                    for repre in representation[model_name]:
                        if len(model_result[(m==model_result.model) & 
                                            (da==model_result.method_da) & 
                                            (da_sel==[str(model_result.lambda_hyper_1), str(model_result.lambda_hyper_2)]) &
                                            (pre == model_result.preprocesamiento) & 
                                            (repre==model_result.representation)])==0:
                            f.writelines(["python ", path_train_a, " ",
                                               path_test_a, " ",
                                               model_name," ",
                                               da," ",
                                               da_sel[0]," ",
                                               da_sel[1]," ",
                                               pre," ",
                                               repre," ",
                                               path_error_a,"\n"])
    f.close()           