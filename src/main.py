import datetime
import matplotlib.pylab as plt

from src.FileManager import FileManager
from src.NeuralNetwork import NeuralNetwork
from src.Timer import Timer


def main():
    # Parse data set
    file_manager = FileManager()
    file_manager.load_file("../Datasets/iris.data")
    train_data = file_manager.get_train_data()
    test_data = file_manager.get_test_data()


    #Train and Plot results of the dataset
    dataset_prediction(train_data,test_data)

    #Plot Hidden Layers v/s Precision Rate
    #plot_hidden_layers_vs_precision_rate(train_data,test_data)

    #Plot mean time of 100 epochs
    #plot_time_vs_epochs(train_data,test_data)

def build_network():
    # Build Neural Network
    neural_network = NeuralNetwork(4)
    neural_network.setRandomLayers(1, 4, 4, 3)
    return neural_network


def dataset_prediction(train_data,test_data):
    #Build neural network
    neural_network = build_network()

    # Train Network
    neural_network.train(1000, train_data, test_data)

    # Assert Ratio
    print("Test data ratio: {0}".format(neural_network.getGuessRatio(test_data)))

    # Plot Data Info
    neural_network.plotErrorData()

def plot_hidden_layers_vs_precision_rate(train_data,test_data):
    hidden_layers = []
    precision_rates = []
    for i in range(10):
        hidden_layers.append(i)
        # Build Neural Network
        neural_network = NeuralNetwork(4)
        neural_network.setRandomLayers(i, 4, 8, 3)

        #Train Network
        neural_network.train(1000,train_data)


        total = float(len(test_data))
        correct_guesses = 0
        for data in test_data:
            guess = neural_network.interp(neural_network.feed(data[0:len(data)]))
            if guess == data[len(data) - 1]:
                correct_guesses += 1
        precision = correct_guesses / total

        precision_rates.append(precision)

    #Plot
    plt.figure()
    plt.title("Hidden Layers v/s Precision Rate", fontsize=20)
    plt.xlabel('Hidden Layers')
    plt.ylabel('Precision Rate')
    plt.plot(hidden_layers, precision_rates)


def plot_time_vs_epochs(train_data,test_data):
    timer = Timer()
    number_epochs = []
    time = []
    total_time = 0
    # Build Neural Network
    neural_network = build_network()

    #200 runs of 100 epochs each
    for i in range (200):
        number_epochs.append(i)
        timer.start()
        for data in test_data:
            for j in range (100):
                neural_network.feed(data)
        this_time = timer.stop()
        time.append(this_time)
        total_time += this_time
    mean_time = total_time / len(number_epochs)

    # Plot
    plt.figure()
    plt.title("Time Taken in 100 Epochs", fontsize=20)
    plt.xlabel('Number of Experiment')
    plt.ylabel('Time')
    plt.scatter(number_epochs,time,color='blue')
    plt.axhline(y=mean_time, color='r', linestyle='-')
    plt.show()



def plot_learning_rate_vs_precision():
    pass



if __name__ == '__main__':
    main()
