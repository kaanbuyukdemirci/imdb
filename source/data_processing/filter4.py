import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter4(chunk_size:int=10**6)->None:
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
        'averageRating']

    Same as filter4_fail.py, but this time we only consider 'language'. If the movie
    has an english title (doesn't matter if it is the original one or not), we keep it.

    This step isn't necessary for the rest of the filters to work, but it helps with the runtime.
    """
    
    # read and write
    read_location_1 = ".\\data\\raw\\"
    read_name_1 = "title.akas.tsv.gz"
    read_location_2 = ".\\data\\processed\\"
    read_name_2 = "filter3_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter4_output.csv"
    
    # read the first file
    with pd.read_csv(filepath_or_buffer=read_location_1+read_name_1, sep='\t', header=0, index_col='titleId', 
                     usecols=['titleId', 'language'], na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn') as context_manager_1:
        
        # to write or append
        first_time = True
        
        for chunk_1 in context_manager_1:
            # drop na
            chunk_1.dropna(axis='index', how='any', inplace=True)
            
            # change the tconst col to drop first 2 tt so that you can change the dtype of 'tconst' column (index)
            chunk_1.set_index(chunk_1.index.str.lstrip('tt').astype(np.int32), drop=True, 
                              append=False, inplace=True, verify_integrity=False)
            
            # now, filter
            filt_1 = chunk_1['language'].str.lower() == 'en'
            chunk_1 = chunk_1[filt_1]
            filt_2 = ~chunk_1.index.duplicated(keep='first')
            chunk_1 = chunk_1[filt_2]
            
            # drop language column, since all are en anyways
            chunk_1.drop(labels=['language'], axis='columns', inplace=True)
            
            # read the second file so that you can merge them together
            with pd.read_csv(filepath_or_buffer=read_location_2+read_name_2, sep=',', header=0, index_col='titleId', 
                     na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype={'releaseYear':np.int16, 'runTime':np.int16}) as context_manager_2:
                
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
    filter4(chunk_size=10**6)