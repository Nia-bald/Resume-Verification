import mnist_loader
import neural_network


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = neural_network.Network([784, 30, 10])

net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
