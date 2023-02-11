import pandas as pd
import numpy as np
import torch

def filter10()->None:
    """
    currently the last filter. We form the last dataset 'dataset.csv', 'dataset.npy', and 'dataset.pt'

    do:
        1) save as csv
        2) convert into array
        3) do dimensionality reduction
        4) save as npy and pt
    """
    
    read_location = ".\\data\\processed\\"
    read_name = "filter9_output.csv"
    write_location = ".\\data\\processed\\"
    write_name_csv = "dataset.csv"
    write_name_npy = "dataset.npy"
    write_name_pt = "dataset.pt"
    
    df = pd.read_csv(filepath_or_buffer=read_location+read_name, sep=',', header=0, index_col='titleId', 
                     na_values=['\\N'], compression='infer',on_bad_lines='warn')
    df = df[list(set(df.columns)-set(['averageRating']))+['averageRating']]
    df = df.convert_dtypes()
    old_df = df.copy().round(1)
    
    df = df.astype(np.float32)
    new_df = df.astype(old_df.dtypes).round(1)
    
    #checking whether there was a lossy conversion or not
    columns = df.columns
    filt = (df[columns[0]] == False)
    for column in columns[1:]:
        filt = filt & (df[column] == False)
    
    if np.any(filt):
        print(old_df[filt])
        print(new_df[filt])
        raise ValueError("Lossy dtype conversion.")
    
    # save the final csv
    df.to_csv(path_or_buf=write_location+write_name_csv, sep=',', header=True, 
              index=True, index_label=None, mode='w', compression='infer')
    np.save(write_location+write_name_npy, df.to_numpy(dtype=np.float32))
    torch.save(torch.from_numpy(df.to_numpy(dtype=np.float32)), write_location+write_name_pt)
    
    
if __name__ == "__main__":
    filter10()