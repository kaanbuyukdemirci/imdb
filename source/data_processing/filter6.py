import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter6(chunk_size:int=10**6)->None:
    """
    from
    ['titleId', 'isAdult', 'releaseYear', 'runTime', 'Fantasy', 'Biography',
        'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History', 'Horror',
        'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult', 'Game-Show',
        'Crime', 'Animation', 'Sport', 'Romance', 'Documentary', 'Action',
        'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short', 'Talk-Show',
        'averageRating', 'personId', 'category']
    to
    ['temporary_index', 'isAdult', 'releaseYear', 'runTime', 'Fantasy',
        'Biography', 'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History',
        'Horror', 'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult',
        'Game-Show', 'Crime', 'Animation', 'Sport', 'Romance', 'Documentary',
        'Action', 'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short',
        'Talk-Show', 'averageRating', 'personId', 'category', 'titleId',
        'primaryName', 'birthYear']

    primaryName is the name of the person.
    birthYear is the birth year of the person.
    """
    
    # read and write
    read_location_1 = ".\\data\\raw\\"
    read_name_1 = "name.basics.tsv.gz"
    read_location_2 = ".\\data\\processed\\"
    read_name_2 = "filter5_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter6_output.csv"
    
    # read the first file
    with pd.read_csv(filepath_or_buffer=read_location_1+read_name_1, sep='\t', header=0, index_col=None, 
                     usecols=['nconst', 'primaryName', 'birthYear'], na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype={'nconst':np.string_, 'primaryName':np.string_, 'birthYear':np.string_}) as context_manager_1:
        
        # to write or append
        first_time = True
        
        for chunk_1 in context_manager_1:
            # drop na
            chunk_1 = chunk_1.dropna(axis='index', how='any', inplace=False).astype({'birthYear':np.int16})
            
            # nconst
            chunk_1['nconst'] = chunk_1['nconst'].str.lstrip('nm').astype(np.int32)
            chunk_1.rename(mapper={'nconst':'personId'}, axis='columns', copy=False, inplace=True, errors='warn')
            
            with pd.read_csv(filepath_or_buffer=read_location_2+read_name_2, sep=',', header=0, index_col='titleId', 
                     na_values=['\\N'], chunksize=chunk_size, compression='infer', on_bad_lines='warn', 
                     dtype={'releaseYear':np.int16, 'runTime':np.int16, 'averageRating':np.float16}) as context_manager_2:
                
                for chunk_2 in context_manager_2:
                    # save titleId. it will get lost on merging since you cannot have duplicate index
                    chunk_2['titleId_backup'] = chunk_2.index
                    
                    # merge
                    chunk_3 = chunk_2.merge(right=chunk_1, how='inner', on='personId', copy=False)
                    
                    # recover titleId. but we still cannot set it as index
                    chunk_3.rename(mapper={'titleId_backup':'titleId'}, axis='columns', copy=False, inplace=True, errors='warn')
                    chunk_3.index.names = ['temporary_index']
                    
                    # write (or append)
                    if first_time:
                        first_time = False
                        chunk_3.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                                       index_label=None, mode='w', compression='infer')
                    else:
                        chunk_3.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                                       index_label=None, mode='a', compression='infer')
                    
                    

if __name__ == "__main__":
    filter6(chunk_size=10**6)