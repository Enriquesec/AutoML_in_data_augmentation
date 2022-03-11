from sklearn.metrics import f1_score
import pandas as pd 

var_predictions = ["label"]
def calculate_error(X, y, baselines, name_data):
    f1_scores = {}
    y_predict_pd = {}
    # Next, we are going to calculate the total result
    for label in var_predictions:
        # Get the predictions
        y_predict_pd[label] = baselines[label].predict(X)
        f1_scores["error_"+label] = f1_score(y, y_predict_pd[label], average='macro')
    return list(f1_scores.values())

    # f1_scores = list(f1_scores.values())
    # file = open(path_errors, "w")
    # for i in f1_scores:
    #     file.write(str(i)+ ",")
    
    #file.write(str(sum(f1_scores)/float(len(f1_scores))))
    # file.close()
    
    #pd.DataFrame(y_predict_pd).to_csv(path_result, index=False, header=False)

    
    # return 0
