import pandas as pd
import numpy as np
from beartype import beartype

@beartype
def filter7(chunk_size:int=10**6)->None:
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
    ['temporary_index', 'isAdult', 'releaseYear', 'runTime', 'Fantasy',
        'Biography', 'Mystery', 'Family', 'Reality-TV', 'Comedy', 'History',
        'Horror', 'Adventure', 'Sci-Fi', 'Western', 'Music', 'Adult',
        'Game-Show', 'Crime', 'Animation', 'Sport', 'Romance', 'Documentary',
        'Action', 'News', 'War', 'Drama', 'Thriller', 'Musical', 'Short',
        'Talk-Show', 'averageRating', 'personId', 'category', 'titleId',
        'primaryName', 'birthYear']
        
    but now, we filter people by their names, we only keep the english names.
    the idea is to do what I couldn't do in filter4_fail.py.
    we can then try to filter movies by how many english named people they have.
    """
    
    # read and write
    read_location_1 = ".\\data\\raw\\"
    read_name_1 = "english_names.txt"
    read_location_2 = ".\\data\\processed\\"
    read_name_2 = "filter6_output.csv"
    write_location = ".\\data\\processed\\"
    write_name = "filter7_output.csv"
    
    # read the first file, and get it ready for redex
    names = np.loadtxt(read_location_1+read_name_1, dtype=np.str_)
    names = "("+np.array2string(names, threshold=names.size, separator=' )|(').replace("'",'').replace('[','').replace(']','')+" )"
    
    with pd.read_csv(filepath_or_buffer=read_location_2+read_name_2, sep=',', header=0, index_col='temporary_index', 
                na_values=['\\N'], chunksize=chunk_size, compression='infer', on_bad_lines='warn', 
                dtype={'releaseYear':np.int16, 'birthYear':np.int16, 'runTime':np.int16, 'averageRating':np.float16}) as context_manager:
        
        # to write or append
        first_time = True
        
        for chunk in context_manager:
            # check names
            filt = chunk['primaryName'].str.contains(names, regex=True, case=True)
            chunk = chunk[filt]
            
            # write (or append)
            if first_time:
                first_time = False
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=True, index=True, 
                                index_label=None, mode='w', compression='infer')
            else:
                chunk.to_csv(path_or_buf=write_location+write_name, sep=',', header=False, index=True, 
                                index_label=None, mode='a', compression='infer')
            
                    
                    

if __name__ == "__main__":
    """
    read_location_1 = ".\\data\\raw\\"
    read_name_1 = "english_names.txt"
    names = np.loadtxt(read_location_1+read_name_1, dtype=np.str_)
    print(names[:10])
    names = "("+np.array2string(names, threshold=names.size, separator=' )|(').replace("'",'').replace('[','').replace(']','')+" )"
    print(names[:100])
    
    import re
    word = 'Jimpa Sangpo Bhutia'
    #word = 'Aaron '
    regexp = re.compile(names)
    if regexp.search(word):
        print('matched')
    else:
        print('no')
    #"""
    
    filter7(chunk_size=10**3)