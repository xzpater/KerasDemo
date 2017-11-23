from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import activations, losses, optimizers
import keras.layers.core as core

def deep_neural_net_demo_for_mnist():
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
    model1 = Sequential(name="QuickDemo")
    input_size = 784
    output_size = 10
    hidden_layer1_neurons_count = 100
    hidden_layer2_neurons_count = 50
    hidden_layer1 = Dense(units=hidden_layer1_neurons_count, activation=activations.tanh, use_bias=True,
                          input_shape=(input_size,))
    hidden_layer2 = Dense(units=hidden_layer2_neurons_count, activation=activations.tanh, use_bias=True)
    # hidden_layer3 = core.Dropout(0.05)
    # hidden_layer3 = Dense(units=hidden_layer3_neurons_count, activation=activations.tanh, use_bias=True)
    output_layer = Dense(units=output_size, activation=activations.softmax, use_bias=True,
                         input_dim=hidden_layer1_neurons_count)
    model1.add(hidden_layer1)
    model1.add(hidden_layer2)
    # model1.add(hidden_layer3)
    model1.add(output_layer)

    # Choose Loss Function
    # Choose Optimization Method
    # Choose Evaluation Metric
    model1.compile(optimizer=optimizers.Adamax(), loss=losses.categorical_crossentropy, metrics=["accuracy"])

    # Choose Training Method,Parameters and Stop Condition
    train_batch_size = 60
    train_epochs = 50

    # Proceed Training Iteration
    model1.fit(x_train_data, y_train_data, batch_size=train_batch_size, epochs=train_epochs, verbose=1)

    # Evaluate Test Performance
    score = model1.evaluate(x_test_data, y_test_data, verbose=1)
    print("\n\nTest Accuracy:", score)

    # Plotting Learned Features

    # Saving Model for Future usage


deep_neural_net_demo_for_mnist()
