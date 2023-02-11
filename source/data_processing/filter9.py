import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter9(chunk_size:int=10**6)->None:
    """
    from
    ['isAdult', 'releaseYear', 'runTime', 'Fantasy', 'Biography', 'Mystery',
       'Family', 'Reality-TV', 'Comedy', 'History', 'Horror', 'Adventure',
       'Sci-Fi', 'Western', 'Music', 'Adult', 'Game-Show', 'Crime',
       'Animation', 'Sport', 'Romance', 'Documentary', 'Action', 'News', 'War',
       'Drama', 'Thriller', 'Musical', 'Short', 'Talk-Show', 'averageRating',
       'personId', 'titleId', 'birthYear', 'score_mean', 'score_median',
       'score_var', 'score_std', 'score_max', 'score_min']
    to
    ['Action', 'Crime', 'isAdult', 'Fantasy', 'Adult', 'Musical',
       'Documentary', 'Family', 'Western', 'score_mean', 'Romance', 'Horror',
       'Drama', 'averageRating', 'Sci-Fi', 'History', 'Thriller',
       'releaseYear', 'runTime', 'Sport', 'Short', 'Comedy', 'score_max',
       'Animation', 'News', 'Adventure', 'score_min', 'Game-Show', 'Music',
       'score_var', 'Mystery', 'War', 'Talk-Show', 'score_median', 'Biography',
       'score_std', 'Reality-TV', 'birthYearMean', 'birthYearMedian',
       'birthYearVar', 'birthYearStd', 'birthYearMax', 'birthYearMin']


    keep chunk_size as high as possible, so that mean and stuff is found much more correctly. 10^6 is pretty enough.
    this is also needed to make sure that we have unique titleIds.

    I could have done much more here:
        extract info about individual people.
        filter movies by english better, like require at least 2 english named people.

    you can add filter 7.1 to implement these before this filter.
    """
    
    # read and write
    read_location = ".\\data\\processed\\"
    read_name = "filter8_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter9_output.csv"
    
    # read the first file
    with pd.read_csv(filepath_or_buffer=read_location+read_name, sep=',', header=0, index_col='temporary_index', 
                     na_values=['\\N'], chunksize=chunk_size, compression='infer',
                     on_bad_lines='warn', dtype={'nconst':np.string_, 'primaryName':np.string_, 'birthYear':np.string_}) as context_manager:
        
        # to write or append
        first_time = True
        
        for chunk in context_manager:
            # drop some unused columns and na
            chunk.drop(labels = ['personId'], axis='columns', inplace=True)
            chunk = chunk.dropna(axis='index', how='any', inplace=False)
            
            # choose the columns to do calculations on
            operation_columns = ['birthYear'] + ['titleId']
            other_columns = list(set(chunk.columns)-set(operation_columns)) + ['titleId']
            chunk_operation = chunk[operation_columns].astype(np.int64)
            chunk_other = chunk[other_columns].drop_duplicates(subset=['titleId']).set_index('titleId')
            
            # groupby titleId
            chunk_operation = chunk_operation.groupby(by='titleId', axis='index', as_index=True)
            
            # extract new values. these are the new columns to add.
            means = chunk_operation.mean(numeric_only=True)
            means.rename(columns={'birthYear': 'birthYearMean'}, inplace=True)
            medians = chunk_operation.median(numeric_only=True)
            medians.rename(columns={'birthYear': 'birthYearMedian'}, inplace=True)
            vars = chunk_operation.var(numeric_only=True)
            vars.rename(columns={'birthYear': 'birthYearVar'}, inplace=True)
            stds = chunk_operation.std(numeric_only=True)
            stds.rename(columns={'birthYear': 'birthYearStd'}, inplace=True)
            maxs = chunk_operation.max(numeric_only=True)
            maxs.rename(columns={'birthYear': 'birthYearMax'}, inplace=True)
            mins = chunk_operation.min(numeric_only=True)
            mins.rename(columns={'birthYear': 'birthYearMin'}, inplace=True)
            
            # now merge them
            chunk_other = chunk_other.merge(right=means, how='inner', on='titleId', copy=False)
            chunk_other = chunk_other.merge(right=medians, how='inner', on='titleId', copy=False)
            chunk_other = chunk_other.merge(right=vars, how='inner', on='titleId', copy=False)
            chunk_other = chunk_other.merge(right=stds, how='inner', on='titleId', copy=False)
            chunk_other = chunk_other.merge(right=maxs, how='inner', on='titleId', copy=False)
            chunk_other = chunk_other.merge(right=mins, how='inner', on='titleId', copy=False)

            # std and var needs more than 1 sample, so there are some NAs
            chunk_other.dropna(axis='index', how='any', inplace=True) 
            
            # write (or append)
            if first_time:
                first_time = False
                chunk_other.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                                index_label=None, mode='w', compression='infer')
            else:
                chunk_other.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                                index_label=None, mode='a', compression='infer')

if __name__ == "__main__":
    filter9(chunk_size=10**6)