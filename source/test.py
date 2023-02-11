import pandas as pd
import numpy as np

read_location = ".\\data\\processed\\"
read_name = "filter9_output.csv"
df = pd.read_csv(filepath_or_buffer=read_location+read_name, sep=',', header=0, index_col=0, 
                     na_values=['\\N'], compression='infer',
                     on_bad_lines='warn')

print(df.columns)
print(df.shape[0])