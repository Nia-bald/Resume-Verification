import numpy as np
import random


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # np.random.rand(a, b) creates a numpy array of dimension a, each element of dimension is a list containing b
        # elements.
        # self.biases itself is a list
        # each element of this self.biases list represents a layer (numpy array), since each neuron has
        # 1 bias associated with it each layer has the no. of biases = no. of neurons
        self.weight = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
        # sizes[a:] means a elements from front is removed
        # sizes[:-a] means a elements from end are removed
        # here each neurons will have no. of weights = no. of neurons in previous layer

    def feed_forward(self, a):
        """ Each iteration of the for loop gives you the output of the current layer, which
            is to be put into function of weight and biases given by sigmoid(np.dot(w,a) +b)
            to get output of the next layer. Also each iteration takes weights and biases from a connection between two
            layers. for eg iteration 1 will take weights and biases from layer 1 and 2, iteration 2 from 2 and 3 and so
            on, until we get the final output.
            So entire for loop takes you from input straight to output vector of 10 rows.
            b, w are elements of list ,self.biases and self.weights, were we already know
            each elements of list self.biases and self.weight represents a numpy array
            also a and b will have same no. of rows since no. of biases = no. of neurons"""
        for b, w in zip(self.biases, self.weight):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        if test_data:
            test_data = list(test_data)

        n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)

        for j in range(epochs):  # Here each loop is training the Network over an ENTIRE epoch
            """Training data is a list of tuples each element of tuple a numpy array of input and output,
             output is the desired output, and input is the input that we put in our neural network to train it
             random.shuffle(training_data) shuffles the training data list so to make sure we are not training our
             network in the same way again and again for each epoch, this is a surface level understanding but I am not 
             able to truly understand what is happening really. 3b1b maybe?"""
            random.shuffle(training_data)
            """mini_batches is a List whose elements are a List, list of elements for a mini batch, mini batch which
                is a list of tuples containing input and desired output"""
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                """   Naming of this method is a bit flawed, it does NOT UPDATE the MINI BATCH, it UPDATES WEIGHTS
                    and BIASES USING that MINI BATCH    """
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(" Epoch {0} : {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """ Here nabla b and nabla w has the same shape as that of weights and biases with all elements set to zero"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weight]

        """ OK let me break down the code below. Each iteration of the for loop calculates (d/dv C(xi)) for
            a particular xi, for all weights and biases (here "v" represents weights and biases, from here I will keep
            using it for weights and biases)and adds it to nabla_v doing this for all values of xi gives you the 
            derivative part of last expression of your whiteboard 
            """

        a = [x for x, y in mini_batch]
        b = [y for x, y in mini_batch]



        delta_nabla_b, delta_nabla_w = self.back_prop(a, b)

        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        """ Now that we have nabla w, which represent the summation term on the board, we can just subtract it with
         weights and biases """

        self.weight = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weight, nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]

        """ Thus weights and biases updated for a mini_batch, in the next chapter we will dwell into
            back propagation algorithm which gave us the d/dvC(xi) for a given xi """

    def back_prop(self, x, y):

        # Creates a list with the shape of biases and weight with all micro element zero

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weight]

        # feed forward
        """temporary variable representing activation of a particular layer, here initialized by
        activation of first layer"""
        activation = x

        activations = [x]  # list to store all the activations. layer by layer
        zs = []  # list to store all z vectors, layer by layer
        for b, w in zip(self.biases, self.weight):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weight[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, a[-i - 1].transpose())

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):

        return output_activations - y

    def evaluate(self, test_data):
        """Test_data is a list of tuples, each tuple contains two elements, one element is a numpy array of input
            and the other is just an integer which shows which index of the final numpy array we want to be 1.
            here self.feed forward gives a numpy array of one dimension containing 10 elements np.argmax gives the
            index of the element which has the highest output after feed_forward is executed that is index of max output
            of the final layer.
            """

        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for x, y in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
