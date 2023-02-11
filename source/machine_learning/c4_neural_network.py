import numpy as np
import torch
from torch.utils.data import DataLoader
from beartype import beartype

try: from c3_neural_network import IMDbDataset, SimpleNN
except ImportError: from .c3_neural_network import IMDbDataset, SimpleNN

try: from plot_config import config
except ImportError: from .plot_config import config
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


@beartype
def neural_network(startOver:bool=True, learning_rate:float=0.001, lamb_regularization_parameter:float=0.001, 
                   batch_size:int=32, number_of_epochs:int=100):
    """
    Parameters
    ----------
    startOver : bool, optional, by default True
        Whether the network will start the traning over or continue the training from the last trained model. 
    learning_rate : int, optional, by default 0.001
        The learning rate.
    lamb_regularization_parameter : float, optional, by default 0.001
        L2 regularization parameter
    batch_size : int, optional, by default 32
        Mini-batch size for training.
    number_of_epochs : int, optional, by default 100
        Number of epochs.

    Returns
    -------
    None
        Just prints out some graphs.
    """
    config()

    # hyperparameters and some other parameters
    startOver = startOver
    learning_rate = learning_rate
    lamb_regularization_parameter = lamb_regularization_parameter
    batch_size = batch_size
    number_of_epochs = number_of_epochs

    # network and data
    real_location = ".\\source\\machine_learning\\cache\\"
    training_data = np.load(real_location+'training_data.npy')
    tarining_label = np.load(real_location+'training_label.npy')
    validation_data = np.load(real_location+'validation_data.npy')
    validation_label = np.load(real_location+'validation_label.npy')

    training_dataset = IMDbDataset(training_data, tarining_label)
    validation_dataset = IMDbDataset(validation_data, validation_label)
    dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    network = SimpleNN(in_shape=training_data.shape[1], learning_rate=learning_rate, lamb=lamb_regularization_parameter)

    del training_data, tarining_label, validation_data, validation_label

    # some functions
    def training():
        for batch_idx, batch in enumerate(dataloader):
            data, label = batch
            network.train_one_step(data, label)

    def eval():
        network.eval()
        training_cost = network.cost_func(network.forward(training_dataset.data), training_dataset.label).detach().cpu().numpy().flatten()[0]
        validation_cost = network.cost_func(network.forward(validation_dataset.data), validation_dataset.label).detach().cpu().numpy().flatten()[0]
        gradient_magnitude = network.gradient_magnitude()
        network.eval()
        return training_cost, validation_cost, gradient_magnitude

    # train and eval
    cache_path = ".\\source\\machine_learning\\cache\\"
    cache_name = "state_dict.pt"
    eval_freq = max(1, int(number_of_epochs//100))

    if startOver:
        pass
    else:
        network.load_state_dict(torch.load(cache_path+cache_name, map_location=network.device))

    training_costs = []
    validation_costs = []
    gradient_magnitudes = []
    for epoch_idx in range(number_of_epochs):
        print(f"{epoch_idx+1}/{number_of_epochs}", end="\r")
        training()
        if epoch_idx % eval_freq == 0:
            training_cost, validation_cost, gradient_magnitude = eval()
            training_costs.append(training_cost)
            validation_costs.append(validation_cost)
            gradient_magnitudes.append(gradient_magnitude)

    # save state dic
    torch.save(network, cache_path+cache_name)

    # results - plots (1)
    x = np.arange(1,int(number_of_epochs//eval_freq)+1)*eval_freq
    y1 = training_costs
    y2 = validation_costs
    y3 = gradient_magnitudes
    y4 = np.ones(x.size)*learning_rate

    plt.figure(0, figsize=(14,6))
    plt.clf()

    plt.subplot(121)
    plt.plot(x, y1, linewidth=3)
    plt.plot(x, y2, linewidth=3)
    plt.legend(["Training", "Validation"])
    plt.title("Training & Validation Loss vs Epoch")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)

    plt.subplot(122)
    plt.plot(x, y3*y4, linewidth=3)
    #plt.legend(["Training", "Validation"])
    plt.title("The magnitude of the gradient times the learning rate vs Epoch")
    plt.xlabel("epoch")
    plt.ylabel("G*lr")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)

    plt.tight_layout()

    # results - plots (2)
    training_predictions = network.forward(training_dataset.data).detach().cpu().numpy().flatten()
    validation_predictions = network.forward(validation_dataset.data).detach().cpu().numpy().flatten()
    training_labels = training_dataset.label.detach().cpu().numpy().flatten()
    validation_labels = validation_dataset.label.detach().cpu().numpy().flatten()
    sort1 = np.argsort(training_labels)
    sort2 = np.argsort(validation_labels)
    y1 = training_predictions[sort1]
    y2 = training_labels[sort1]
    y3 = validation_predictions[sort2]
    y4 = validation_labels[sort2]

    plt.figure(1, figsize=(14,6))
    plt.clf()

    plt.subplot(121)
    plt.plot(y1, linewidth=3)
    plt.plot(y2, linewidth=3)
    plt.legend(["Prediction", "Label"])
    plt.title(f"Training Predictions vs Labels (MSE={round(training_costs[-1],2)})")
    plt.xlabel("sample")
    plt.ylabel("average imdb score")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)

    plt.subplot(122)
    plt.plot(y3, linewidth=3)
    plt.plot(y4, linewidth=3)
    plt.legend(["Prediction", "Label"])
    plt.title(f"Validation Predictions vs Labels (MSE={round(validation_costs[-1],2)})")
    plt.xlabel("sample")
    plt.ylabel("average imdb score")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    neural_network()
