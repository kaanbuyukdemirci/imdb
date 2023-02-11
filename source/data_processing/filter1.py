import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter1(lower_limit:int, upper_limit:int, chunk_size:int=10**6)->None:
    """
    from 
    [] 
    to 
    ['titleId', 'isAdult', 'releaseYear', 'runTime', 'genres']

    titleId is the unique identifier of a movie.
    isAdult is whether the movie is a adult movie or not.
    releaseYear is the release year of the movie.
    runtime is the runtime of the movie.
    genres is the list if genres that the movie belongs to.
    """
        
    # read and write
    read_location = ".\\data\\raw\\" #".\\..\\..\\data\\raw\\"
    read_name = "title.basics.tsv.gz"
    write_location = ".\\data\\processed\\" #".\\..\\..\\data\\processed\\"
    write_name = "filter1_output.csv"
    cols = ['tconst', 'titleType', 'startYear', 'isAdult', 'runtimeMinutes', 'genres'] # columns to get from title.basics.tsv.gz
    
    with pd.read_csv(filepath_or_buffer=read_location+read_name, sep='\t', header=0, index_col='tconst', 
                     usecols=cols, na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype=np.string_) as context_manager:
        
        # to write or append
        first_time = True
        
        for chunk in context_manager:
            # drop na so that you can change the dtype of 'startYear' column.
            chunk = chunk.dropna(axis='index', how='any', inplace=False).astype({'startYear':np.int16, 
                                                                                 'isAdult':np.int8, 
                                                                                 'runtimeMinutes':np.int16}).astype({'isAdult':np.bool_})
            
            # change the tconst col to drop first 2 tt so that you can change the dtype of 'tconst' column (index)
            chunk.set_index(chunk.index.str.lstrip('tt').astype(np.int32), drop=True, 
                            append=False, inplace=True, verify_integrity=False)
            
            # rename the columns
            chunk.index.names = ['titleId']
            chunk.rename(mapper={'startYear':'releaseYear', 'runtimeMinutes':'runTime'}, 
                         axis='columns', copy=False, inplace=True, errors='raise')
            
            # filter 1.1 (movies from titleType)
            # print(chunk['titleType'].unique()) # to see the labels
            filt1 = chunk['titleType']=='movie'
            chunk = chunk[filt1].loc[:, chunk.columns[chunk.columns!='titleType']]
        
            # filter 1.2 (release years from releaseYear)
            filt2 = (chunk['releaseYear']>lower_limit) & (chunk['releaseYear']<upper_limit)
            chunk = chunk[filt2]
        
            # write (or append)
            if first_time:
                first_time = False
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                             index_label=None, mode='w', compression='infer')
            else:
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                             index_label=None, mode='a', compression='infer')
            
    return None

if __name__ == "__main__":
    import os
    print(os.getcwd())
    filter1(lower_limit=2000, upper_limit=2025, chunk_size=10**6)