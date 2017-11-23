from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import activations, losses, optimizers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy


def shallow_neural_net_demo_for_mnist():
    # Loading DataSet
    file_path = "D:\Data\MLData\mnist\mnist.npz"
    (x_train_data, y_train_data), (x_test_data, y_test_data) = mnist.load_data(file_path)

    # DataSet Processing
    num_of_classes = 10
    x_train_data = x_train_data.reshape(60000, 784)
    x_test_data = x_test_data.reshape(10000, 784)
    y_train_data = np_utils.to_categorical(y_train_data, num_classes=num_of_classes)
    y_test_data = np_utils.to_categorical(y_test_data, num_classes=num_of_classes)

    # Create Neural Model
    #   Choose Layers Structure
    #       Choose Layer Size and Depth
    #           Input Layer: Size, Normalization
    #           Output Layer: Size, Type, Normalization
    #           Optional Hidden Layer: Size, Depth
    #       Optional Bias Neurons
    #       Choose Activation Function
    #       Choose Optional Regulation
    #       Choose Optional Weight Initialization Method
    model1 = Sequential(name="Demo1")
    input_size = 784
    output_size = 10
    hidden_layer1_neurons_count = 200
    hidden_layer1 = Dense(units=hidden_layer1_neurons_count, activation=activations.tanh, use_bias=True,
                          input_shape=(input_size,))

    output_layer = Dense(units=output_size, activation=activations.softmax, use_bias=True,
                         input_dim=hidden_layer1_neurons_count)

    model1.add(hidden_layer1)
    model1.add(output_layer)

    # Choose Loss Function
    # Choose Optimization Method
    # Choose Evaluation Metric
    model1.compile(optimizer=optimizers.Adamax(), loss=losses.categorical_crossentropy, metrics=["accuracy"])

    # Choose Training Method,Parameters and Stop Condition
    train_batch_size = 60
    train_epochs = 30

    # Proceed Training Iteration
    model1.fit(x_train_data, y_train_data, batch_size=train_batch_size, epochs=train_epochs, verbose=1)

    # Evaluate Test Performance
    score = model1.evaluate(x_test_data, y_test_data, verbose=1)
    print("\n\nTest Accuracy:", score)

    # Plotting manual scripts of digits
    width = 10
    height = hidden_layer1_neurons_count // width + (hidden_layer1_neurons_count % width)
    fig2 = plt.figure("数字手写体样例")
    for pixel_index in range(0, hidden_layer1_neurons_count):
        ax1 = fig2.add_subplot(height, width, pixel_index + 1)
        ax1.axis("off")
        ax1.imshow(numpy.reshape(x_train_data[pixel_index], (28, 28)), cmap=cm.Greys_r)
    plt.savefig("manual_digits.png", dpi=300)

    # Plotting Learned Features
    fig = plt.figure("训练出来的NN权重分布")
    weights = model1.layers[0].get_weights()
    w = weights[0].T


    for neuron in range(hidden_layer1_neurons_count):
        ax = fig.add_subplot(height, width, neuron + 1)
        ax.axis("off")
        ax.imshow(numpy.reshape(w[neuron], (28, 28)), cmap=cm.Greys_r)
    plt.savefig("neuron_images.png", dpi=300)
    plt.show()
    # Saving Model for Future usage


shallow_neural_net_demo_for_mnist()
