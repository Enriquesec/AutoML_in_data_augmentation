import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import pandas as pd
import nltk
nltk.download('omw-1.4')
def noise_injection(lambda_hyper):
    aug = naw.RandomWordAug(action=lambda_hyper[0], aug_p=float(lambda_hyper[1]))
    return aug

def replacing_synonym(lambda_hyper):
    aug = naw.SynonymAug(aug_src='wordnet',  lang="spa", aug_p=float(lambda_hyper[1]))
    return aug

def back_translation(lambda_hyper):
    aug = naw.BackTranslationAug(
    from_model_name=lambda_hyper[0], 
    to_model_name=lambda_hyper[1])
    return aug

def replacing_embeddings(lambda_hyper):
    if lambda_hyper[0] == "fasttext":
        model_path_embs = '../../fasttext-sbwc.3.6.e20.vec'
    elif lambda_hyper[1] == "glove":
        model_path_embs = "../../glove-sbwc.i25.vec"
    elif lambda_hyper[2] == "word2vect":
        model_path_embs  = "../../SBW-vectors-300-min5.vec"
    aug = naw.WordEmbsAug(
        model_type=lambda_hyper[0], model_path=model_path_embs,
        action=lambda_hyper[1])
    return aug 

# def expresion_regulares_spa():
	# FALTA! 
	
def selecction_method_aug(taks_augment, lambda_hyper):
    if taks_augment=="noise_injection":
        aug = noise_injection(lambda_hyper)
    elif taks_augment=="replacing_synonym":
        aug = replacing_synonym(lambda_hyper)
    elif taks_augment=="back_translation":
        aug = back_translation(lambda_hyper)
    elif taks_augment=="replacing_embeddings":
    	aug = replacing_embeddings(lambda_hyper)
    else:
        print("Selection method da incorrect")
        exit()
    return aug

def data_augment(df, col_int, taks_augment, lambda_hyper):
    aug = selecction_method_aug(taks_augment, lambda_hyper)
    df_augment = []
    for i in range(df.shape[0]):
        df_augment.append(aug.augment(df.loc[i,col_int][0:500], n=1))
    df_aug = df.copy()
    df_aug[col_int] = df_augment 
    
    df_cont = pd.concat([df, df_aug])
    
    return df_cont.drop_duplicates(), str(df_cont.shape[0])