import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter2(chunk_size:int=10**6)->None:
    """
    from
    ['titleId', 'isAdult', 'releaseYear', 'runTime', 'genres']
    to
    ['titleId', 'isAdult', 'releaseYear', 'runTime', 'Fantasy', 'Biography',
        'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History', 'Horror',
        'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult', 'Game-Show',
        'Crime', 'Animation', 'Sport', 'Romance', 'Documentary', 'Action',
        'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short', 'Talk-Show']
        
    basically onehot repserentation of genres
    """
    
    # read and write
    read_location = ".\\data\\processed\\"
    read_name = "filter1_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter2_output.csv"
    
    # learn the genres
    set_of_genres = set()
    with pd.read_csv(filepath_or_buffer=read_location+read_name, sep=',', header=0, index_col='titleId', 
                     usecols=['titleId', 'genres'], na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype=np.string_) as context_manager:
        
        for chunk in context_manager:
            set_of_genres = set_of_genres.union(set([genre for genres in chunk['genres'].values 
                                                     for genre in genres.split(',')]))
    set_of_genres = np.array(list(set_of_genres))
    
    # add a column for each genre
    with pd.read_csv(filepath_or_buffer=read_location+read_name, sep=',', header=0, index_col='titleId', 
                     na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype=np.string_) as context_manager:
        
        # to write or append
        first_time = True
        
        for chunk in context_manager:
            # add the new columns, initiate them as False
            shape = (chunk.shape[0], set_of_genres.shape[0])
            chunk[set_of_genres] = np.zeros(shape, dtype=np.bool_)
            
            # update
            for column_name in set_of_genres:
                chunk[column_name] = chunk['genres'].str.contains(column_name).astype(np.bool_)
            
            # drop genres column
            chunk.drop(labels=['genres'], axis='columns', inplace=True)
            
            # write (or append)
            if first_time:
                first_time = False
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                             index_label=None, mode='w', compression='infer')
            else:
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                             index_label=None, mode='a', compression='infer')

if __name__ == "__main__":
    filter2(chunk_size=10**6)