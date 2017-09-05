from src.FileManager import FileManager
from src.NeuralNetwork import NeuralNetwork


def main():

    #Parse data set
    file_manager = FileManager()
    file_manager.load_file("../Datasets/iris.data")
    train_data = file_manager.get_train_data()
    test_data = file_manager.get_test_data()

    #Build Neural Network
    neural_network = NeuralNetwork(4)
    neural_network.setRandomLayers(1, 4, 4, 3)

    #Train Network
    neural_network.train(1000, train_data,test_data)

    #Assert Ratio
    print("Test data ratio: {0}".format(neural_network.getGuessRatio(test_data)))

    #Plot Data Info
    neural_network.plotErrorData()


def plot_hidden_layers_vs_learning_rate():
    pass

def plot_time_vs_epochs():
    pass

def plot_learning_rate_vs_precision():
    pass



if __name__ == '__main__':
    main()

main()
