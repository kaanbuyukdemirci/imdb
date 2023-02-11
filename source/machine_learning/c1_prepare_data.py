import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import KernelPCA

import matplotlib.colors as mcolors
try: from plot_config import config
except ImportError: from .plot_config import config

def prepare_data():
    config()
    
    # data
    location = "data\\processed\\"
    name_npy = "dataset.npy"
    name_csv = "dataset.csv"
    df = pd.read_csv(filepath_or_buffer=location+name_csv, sep=',', header=0, index_col=0, na_values=['\\N'], 
                    compression='infer', on_bad_lines='warn').convert_dtypes()
    df = df.sample(frac = 1) # shuffle

    # imdb score dist before
    average_imdbs = df['averageRating']
    plt.figure(0)
    plt.clf()
    plt.hist(average_imdbs,bins=[0,1,2,3,4,5,6,7,8,9,10], density=True)
    plt.title(f"Probability density function of the IMDb scores (before clipping) ({df.shape[0]} samples)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.show()

    # eliminate some of the samples so we have better class balance
    how_many_samples_max = 1000
    frames = []
    for i in range(10):
        frames.append(df[(df['averageRating']>=i) & (df['averageRating']<i+1)][:how_many_samples_max])
    df = pd.concat(frames)

    # imdb score dist after
    average_imdbs = df['averageRating']
    plt.figure(0)
    plt.clf()
    plt.hist(average_imdbs,bins=[0,1,2,3,4,5,6,7,8,9,10], density=True)
    plt.title(f"Probability density function of the IMDb scores (after clipping) ({df.shape[0]} samples)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.show()

    # to numpy
    dataset = df.to_numpy(dtype=np.float32)
    np.random.shuffle(dataset)
    data = dataset[:,:-1]
    labels = dataset[:,-1].reshape(-1, 1)

    validation_rate = 0.1
    seperation_index = int(validation_rate*labels.shape[0])
    training_data = data[:-seperation_index,:]
    tarining_label = labels[:-seperation_index,:]
    validation_data = data[-seperation_index:,:]
    validation_label = labels[-seperation_index:,:]

    del df, average_imdbs, frames, dataset, data, labels

    # standardize
    mean = np.mean(training_data, axis=0)
    std = np.std(training_data, axis=0)
    std[std==0] = 1
    training_data = (training_data - mean) / std
    validation_data = (validation_data - mean) / std
    #training_data = np.nan_to_num(training_data)
    #validation_data = np.nan_to_num(validation_data)

    # dimensionality reduction
    #pca_model = KernelPCA(n_components=5, kernel='rbf')
    #pca_model.fit(training_data)
    #training_data = pca_model.transform(training_data)
    #validation_data = pca_model.transform(validation_data)

    # save
    write_location = ".\\source\\machine_learning\\cache\\"
    np.save(write_location+"training_data.npy", training_data)
    np.save(write_location+"training_label.npy", tarining_label)
    np.save(write_location+"validation_data.npy", validation_data)
    np.save(write_location+"validation_label.npy", validation_label)

if __name__ == "__main__":
    prepare_data()