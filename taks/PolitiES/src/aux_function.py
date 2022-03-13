import random
import numpy as np 
import csv
import pandas as pd
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