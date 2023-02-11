import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter8(chunk_size:int=10**6)->None:
    """
    from
    ['temporary_index', 'isAdult', 'releaseYear', 'runTime', 'Fantasy',
        'Biography', 'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History',
        'Horror', 'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult',
        'Game-Show', 'Crime', 'Animation', 'Sport', 'Romance', 'Documentary',
        'Action', 'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short',
        'Talk-Show', 'averageRating', 'personId', 'category', 'titleId',
        'primaryName', 'birthYear']
    to
    ['isAdult', 'releaseYear', 'runTime', 'Fantasy', 'Biography', 'Mystery',
       'Family', 'Reality-TV', 'Comedy', 'History', 'Horror', 'Adventure',
       'Sci-Fi', 'Western', 'Music', 'Adult', 'Game-Show', 'Crime',
       'Animation', 'Sport', 'Romance', 'Documentary', 'Action', 'News', 'War',
       'Drama', 'Thriller', 'Musical', 'Short', 'Talk-Show', 'averageRating',
       'personId', 'titleId', 'birthYear', 'score_mean', 'score_median',
       'score_var', 'score_std', 'score_max', 'score_min']
    """
    
    # read and write
    read_location = ".\\data\\processed\\"
    read_name = "filter7_better_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter8_output.csv"
    
    # read the first file
    with pd.read_csv(filepath_or_buffer=read_location+read_name, sep=',', header=0, index_col='temporary_index', 
                     na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype={'nconst':np.string_, 'primaryName':np.string_, 'birthYear':np.string_}) as context_manager:
        # to write or append
        first_time = True
        
        for chunk in context_manager:
            # drop some unused columns and na
            chunk.drop(labels = ['category', 'primaryName'], axis='columns', inplace=True)
            chunk = chunk.dropna(axis='index', how='any', inplace=False)
            
            # detect every unique movie
            movies = chunk.drop_duplicates(subset=["titleId"])
            
            # open up new columns to fill later
            chunk['score_mean'] = np.zeros(chunk.shape[0])
            chunk['score_median'] = np.zeros(chunk.shape[0])
            chunk['score_var'] = np.zeros(chunk.shape[0])
            chunk['score_std'] = np.zeros(chunk.shape[0])
            chunk['score_max'] = np.zeros(chunk.shape[0])
            chunk['score_min'] = np.zeros(chunk.shape[0])
            
            
            # iterate through movies
            #count = movies['titleId'].shape[0] # =================
            #counter = 0
            for movie in movies['titleId']:
                #counter += 1
                #print(f"{counter}/{count}", end="\r")
                # filter out the movies that came after that movie
                release_year = movies[movies['titleId']==movie]['releaseYear'].to_numpy().flatten()[0]
                filtered_chunk = chunk[chunk['releaseYear'].to_numpy().flatten()<release_year]
                
                # filter out the other casts
                filt = np.zeros(filtered_chunk.shape[0]).astype(np.bool_)
                for cast in movies[movies['titleId']==movie]['personId']:
                    filt = filt | (filtered_chunk['personId']==cast).to_numpy().flatten()
                filtered_chunk = filtered_chunk[filt]
                
                # do
                mean = filtered_chunk['averageRating'].mean()
                median = filtered_chunk['averageRating'].median()
                var = filtered_chunk['averageRating'].var()
                std = filtered_chunk['averageRating'].std()
                max_ = filtered_chunk['averageRating'].max()
                min_ = filtered_chunk['averageRating'].min()
                
                # add
                chunk.loc[chunk['titleId']==movie,'score_mean'] = mean
                chunk.loc[chunk['titleId']==movie,'score_median'] = median
                chunk.loc[chunk['titleId']==movie,'score_var'] = var
                chunk.loc[chunk['titleId']==movie,'score_std'] = std
                chunk.loc[chunk['titleId']==movie,'score_max'] = max_
                chunk.loc[chunk['titleId']==movie,'score_min'] = min_

            chunk.dropna(axis='index', how='any', inplace=True) 
            
            # write (or append)
            if first_time:
                first_time = False
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                                index_label=None, mode='w', compression='infer')
            else:
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                                index_label=None, mode='a', compression='infer')
                
                
if __name__ == "__main__":
    filter8(chunk_size=10**6)
                
            
            