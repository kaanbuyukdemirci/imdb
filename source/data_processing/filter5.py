import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter5(chunk_size:int=10**6)->None:
    """
    from
    ['titleId', 'isAdult', 'releaseYear', 'runTime', 'Fantasy', 'Biography',
        'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History', 'Horror',
        'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult', 'Game-Show',
        'Crime', 'Animation', 'Sport', 'Romance', 'Documentary', 'Action',
        'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short', 'Talk-Show',
        'averageRating']
    to
    ['titleId', 'isAdult', 'releaseYear', 'runTime', 'Fantasy', 'Biography',
        'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History', 'Horror',
        'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult', 'Game-Show',
        'Crime', 'Animation', 'Sport', 'Romance', 'Documentary', 'Action',
        'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short', 'Talk-Show',
        'averageRating', 'personId', 'category']

    personId is the unique identifier for a person (actor, director, etc.)
    category is that person's job (actor, director, etc.)

    each row used to have a unique titleId, because our data was in terms of movies.
    Now they don't, because our data now is in terms of (movie, person) pairs.
    So now, each row has a unique (titleId, personId) pair.
    """
    
    # read and write
    read_location_1 = ".\\data\\raw\\"
    read_name_1 = "title.principals.tsv.gz"
    read_location_2 = ".\\data\\processed\\"
    read_name_2 = "filter4_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter5_output.csv"
    
    # read the first file
    with pd.read_csv(filepath_or_buffer=read_location_1+read_name_1, sep='\t', header=0, index_col='tconst', 
                     usecols=['category', 'nconst', 'tconst'], na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype={'tconst':np.string_, 'averageRating':np.float16}) as context_manager_1:
        
        # to write or append
        first_time = True
        
        for chunk_1 in context_manager_1:
            # drop na
            chunk_1.dropna(axis='index', how='any', inplace=True)
            
            # change the tconst col to drop first 2 tt so that you can change the dtype of 'tconst' column (index)
            chunk_1.set_index(chunk_1.index.str.lstrip('tt').astype(np.int32), drop=True, 
                              append=False, inplace=True, verify_integrity=False)
            chunk_1.index.names = ['titleId']
            
            # nconst
            chunk_1['nconst'] = chunk_1['nconst'].str.lstrip('nm').astype(np.int32)
            chunk_1.rename(mapper={'nconst':'personId'}, axis='columns', copy=False, inplace=True, errors='warn')
            
            with pd.read_csv(filepath_or_buffer=read_location_2+read_name_2, sep=',', header=0, index_col='titleId', 
                     na_values=['\\N'], chunksize=chunk_size, compression='infer', on_bad_lines='warn', 
                     dtype={'releaseYear':np.int16, 'runTime':np.int16, 'averageRating':np.float16}) as context_manager_2:
                
                for chunk_2 in context_manager_2:
                    # merge
                    chunk_3 = chunk_2.merge(right=chunk_1, how='inner', on='titleId', copy=False)
                    
                    # write (or append)
                    if first_time:
                        first_time = False
                        chunk_3.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                                       index_label=None, mode='w', compression='infer')
                    else:
                        chunk_3.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                                       index_label=None, mode='a', compression='infer')
                    

if __name__ == "__main__":
    filter5(chunk_size=10**6)