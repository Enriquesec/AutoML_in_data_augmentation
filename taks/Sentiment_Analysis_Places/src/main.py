from aux_function import *
from preprocessing_data_model_sa import *
from modeling_and_erros import * 
import sys
import csv


if __name__ == '__main__':
    if len(sys.argv) != 10:
        print("Wrong Configurations Number!!!")
        exit(1)
    else:
        path_train = sys.argv[1] 
        path_test = sys.argv[2] 
        model_name = [sys.argv[3]]
        method_da = [sys.argv[4]]
        lambda_hyper=[sys.argv[5],sys.argv[6]] 
        preprocess=[sys.argv[7]] 
        representation=[sys.argv[8]] 
        path_errors=sys.argv[9]
    
    baseline_polities(path_train, path_test, model_name, method_da, lambda_hyper, preprocess, 
    representation, path_errors)
