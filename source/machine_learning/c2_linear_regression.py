import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
try: from plot_config import config
except ImportError: from .plot_config import config

def linear_regression():
    config()

    # load training and validation
    real_location = ".\\source\\machine_learning\\cache\\"
    training_data = np.load(real_location+'training_data.npy')
    tarining_label = np.load(real_location+'training_label.npy')
    validation_data = np.load(real_location+'validation_data.npy')
    validation_label = np.load(real_location+'validation_label.npy')

    # add in the bias
    training_data = np.c_[training_data, np.ones(training_data.shape[0]).reshape((-1, 1))]
    validation_data = np.c_[validation_data, np.ones(validation_data.shape[0]).reshape((-1, 1))]

    # parameter vector
    I = np.diag(np.ones(training_data.shape[1]))
    lamb = 1
    parameter_vector = np.linalg.inv(training_data.T @ training_data + lamb*I) @ training_data.T @ tarining_label
    #print(np.linalg.norm(parameter_vector))

    # training and validation predictions
    training_predictions = training_data @ parameter_vector
    training_MSE = ((training_predictions - tarining_label).T @ (training_predictions - tarining_label) / training_predictions.shape[0])[0,0]
    tarining_label = tarining_label.flatten()
    training_predictions = training_predictions.flatten()

    validation_predictions = validation_data @ parameter_vector
    validation_MSE = ((validation_predictions - validation_label).T @ (validation_predictions - validation_label) / validation_predictions.shape[0])[0,0]
    validation_label = validation_label.flatten()
    validation_predictions = validation_predictions.flatten()

    # results
    sort1 = np.argsort(tarining_label)
    sort2 = np.argsort(validation_label)
    y1 = training_predictions[sort1]
    y2 = tarining_label[sort1]
    y3 = validation_predictions[sort2]
    y4 = validation_label[sort2]

    plt.figure(1, figsize=(14,6))
    plt.clf()

    plt.subplot(121)
    plt.plot(y1, linewidth=3)
    plt.plot(y2, linewidth=3)
    plt.legend(["Prediction", "Label"])
    plt.title(f"Training Predictions vs Labels (MSE={round(training_MSE,2)})")
    plt.xlabel("sample")
    plt.ylabel("average imdb score")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)

    plt.subplot(122)
    plt.plot(y3, linewidth=3)
    plt.plot(y4, linewidth=3)
    plt.legend(["Prediction", "Label"])
    plt.title(f"Validation Predictions vs Labels (MSE={round(validation_MSE,2)})")
    plt.xlabel("sample")
    plt.ylabel("average imdb score")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    linear_regression()