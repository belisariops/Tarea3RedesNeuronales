import matplotlib.pylab as plt
import numpy as np

from src.FileManager import FileManager
from src.NeuralNetwork import NeuralNetwork
from src.Timer import Timer
from random import shuffle

from src.GeneticFixedToplogy import GeneticFixedTopology


def main():
    # Parse data set
    file_manager = FileManager()
    file_manager.load_file("../Datasets/iris.data")
    train_data = file_manager.get_train_data()
    test_data = file_manager.get_test_data()
    number_of_epochs = 2000
    # Training data can be shuffled
    # shuffle(train_data)

    """
    Genetic Algorithm (Tarea 3)
    """
    # -------------------------------------------------
    genetic = GeneticFixedTopology(100, 1000)
    best_neural_network = genetic.run()
    genetic.plot_results()
    #plot_genetic_algorithm_time()
    # -------------------------------------------------

    # Train and Plot results of the dataset
    # dataset_prediction(train_data, test_data, number_of_epochs)

    # Plot Hidden Layers v/s Precision Rate
    # plot_hidden_layers_vs_precision_rate(train_data,test_data)

    # Plot mean time of 1000 epochs
    # plot_time_vs_epochs(train_data,test_data)

    # Plot learning rate v/s precision
    # plot_learning_rate_vs_precision(train_data,test_data)


def build_network():
    # Build Neural Network
    neural_network = NeuralNetwork(4)
    neural_network.setRandomLayers(1, 4, 4, 3)
    return neural_network


def dataset_prediction(train_data, test_data, number_of_epochs):
    # Build neural network
    neural_network = build_network()

    # Train Network
    neural_network.train(number_of_epochs, train_data, test_data)

    # Assert Ratio
    print("Test data ratio: {0}".format(neural_network.getGuessRatio(test_data)))
    # Plot Data Info
    neural_network.plotErrorData()


def plot_hidden_layers_vs_precision_rate(train_data, test_data):
    hidden_layers = []
    precision_rates = []
    for i in range(20):
        hidden_layers.append(i)
        # Build Neural Network
        neural_network = NeuralNetwork(4)
        neural_network.setRandomLayers(i, 8, 8, 3)

        # Train Network
        neural_network.train(1000, train_data)

        precision_rates.append(getPrecision(neural_network, test_data))

    # Plot
    plt.figure()
    plt.title("Hidden Layers v/s Precision Rate", fontsize=20)
    plt.xlabel('Hidden Layers')
    plt.ylabel('Precision Rate')
    plt.plot(hidden_layers, precision_rates)
    plt.show()


def plot_time_vs_epochs(train_data, test_data):
    timer = Timer()
    number_epochs = []
    time = []
    total_time = 0
    # Build Neural Network
    neural_network = build_network()

    # 200 runs of 100 epochs each
    for i in range(100):
        number_epochs.append(i)
        timer.start()
        for j in range(1000):
            for data in test_data:
                neural_network.feed(data)
        this_time = timer.stop()
        time.append(this_time)
        total_time += this_time
    mean_time = total_time / len(number_epochs)

    # Plot
    plt.figure()
    plt.title("Time Taken in 1000 Epochs", fontsize=20)
    plt.xlabel('Number of Experiment')
    plt.ylabel('Time (seconds)')
    plt.scatter(number_epochs, time, color='blue')
    plt.axhline(y=mean_time, color='r', linestyle='-')
    plt.show()


def plot_learning_rate_vs_precision(train_data, test_data):
    learning_rates = np.linspace(0.05, 2, 20)
    precision = []
    for rate in learning_rates:
        neural_network = build_network()
        neural_network.setLearningRate(rate)
        neural_network.train(1000, train_data)
        precision.append(getPrecision(neural_network, test_data))

    # Plot
    plt.figure()
    plt.title("Learning Rate v/s Precision", fontsize=20)
    plt.xlabel('Learning Rate')
    plt.ylabel('Precision')
    plt.plot(learning_rates, precision)
    plt.show()


def getPrecision(neural_network, test_data):
    total = float(len(test_data))
    correct_guesses = 0
    for data in test_data:
        guess = neural_network.interp(neural_network.feed(data[0:len(data)]))
        if guess == data[len(data) - 1]:
            correct_guesses += 1
    precision = correct_guesses / total

    return precision


def plot_genetic_algorithm_time():
    timer = Timer()
    experiments = []
    time = []
    total_time = 0
    # Build Neural Network
    genetic_algorithm = GeneticFixedTopology(100, 1000)

    # 20 runs of 1000 generations each
    for i in range(20):
        experiments.append(i)
        timer.start()
        genetic_algorithm.run()
        this_time = timer.stop()
        time.append(this_time)
        genetic_algorithm = GeneticFixedTopology(100, 1000)
        total_time += this_time
    mean_time = total_time / len(experiments)

    # Plot
    plt.figure()
    plt.title("Time Taken in 1000 Generations", fontsize=20)
    plt.xlabel('experimento')
    plt.ylabel('tiempo (segundos)')
    plt.scatter(experiments, time, color='blue')
    plt.axhline(y=mean_time, color='r', linestyle='-')
    plt.show()


if __name__ == '__main__':
    main()
