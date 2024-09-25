from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

TF_VERSION = 1


def classify_digits_v1():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    train_images = x_train.reshape(60000, 784)
    test_images = x_test.reshape(10000, 784)
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Normalize the images to a value between 0 and 1
    x_train, x_test = train_images / 255.0, test_images / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # MNIST provides 60,000 samples in a training data set, 10,000 samples in a test data 
    # set, and 5,000 samples in a "validation" data set. Validation sets are used for 
    # model selection, so you use validation data to select your model, train the model 
    # with the training set, and then evaluate the model using the test data set.

    # The training data, after you "flatten" it to one dimension using the reshape 
    # function, is therefore a tensor of shape [60,000, 784]: 60,000 instances of 
    # 784 numbers that represent each image.

    # Function to visualize what the input data looks like, and pick a random training image
    def display_sample(num):
        #Print the one-hot array of this sample's label
        print(y_train[num]) 

        #Print the label converted back to a number
        label = y_train[num].argmax(axis=0)

        #Reshape the 768 values to a 28x28 image
        image = x_train[num].reshape([28,28])

        plt.title('Sample: %d  Label: %d' % (num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()

    display_sample(1234)

    # Visualize how the data is being fed into the model
    images = x_train[0].reshape([1,784])

    for i in range(1, 500):
        images = np.concatenate((images, x_train[i].reshape([1,784])))

    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

    # While training, assign input_images to the training images and target_labels to the training labels. 
    # While testing, use the test images and test labels instead.
    input_images = tf.placeholder(tf.float32, shape=[None, 784])
    target_labels = tf.placeholder(tf.float32, shape=[None, 10])

    # Deep Neural Network setup.
    # Input layer with one node per input pixel per image, or 784 nodes.
    # Hidden layer with arbitary size, 512.
    # Output layer of 10 values, corresponding to scores for each classification to be fed into softmax.
    """
    hidden_nodes = 512
    input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
    input_biases = tf.Variable(tf.zeros([hidden_nodes]))
    hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
    hidden_biases = tf.Variable(tf.zeros([10]))

    # Feed into hidden layer, which applies the ReLU activation function to the weighted inputs with the learned biases added in as well.
    input_layer = tf.matmul(input_images, input_weights)
    hidden_layer = tf.nn.relu(input_layer + input_biases)

    # Output layer, called digit_weights, multiplies in the learned weights of the hidden layer and adds in the hidden layer's bias term.
    digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases
    """

    # Code for 2 hidden layers
    # Output layer of 10 values, corresponding to scores for each classification to be fed into softmax.
    hidden_nodes_1 = 512
    hidden_nodes_2 = 256
    # Input layer to first hidden layer
    input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes_1]))
    input_biases = tf.Variable(tf.zeros([hidden_nodes_1]))
    hidden_layer_1 = tf.nn.relu(tf.matmul(input_images, input_weights) + input_biases)

    # First hidden layer to second hidden layer
    hidden_weights_1 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2]))
    hidden_biases_1 = tf.Variable(tf.zeros([hidden_nodes_2]))
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, hidden_weights_1) + hidden_biases_1)

    # Second hidden layer to output layer
    output_weights = tf.Variable(tf.truncated_normal([hidden_nodes_2, 10]))
    output_biases = tf.Variable(tf.zeros([10]))
    digit_weights = tf.matmul(hidden_layer_2, output_weights) + output_biases
    

    # Define the loss function for use in measuring progress in gradient descent: cross-entropy, which applies a logarithmic scale to 
    # penalize incorrect classifications much more than ones that are close.
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=target_labels))

    # Gradient descent optimizer with aggressive learning rate. 
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss_function)

    # Train and measure accuracy
    # correct_prediction will look at the output of the neural network (in digit_weights), choose the label with the highest value, 
    # and see if that agrees with the target label given.
    # accuracy then takes the average of all the classifications to produce an overall score for our model's accuracy.
    correct_prediction = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()

    EPOCH = 20
    BATCH_SIZE = 100
    TRAIN_DATASIZE,_ = x_train.shape
    PERIOD = TRAIN_DATASIZE//BATCH_SIZE
    
    for e in range(EPOCH):
        idxs = np.random.permutation(TRAIN_DATASIZE)
        X_random = x_train[idxs]
        Y_random = y_train[idxs]

        for i in range(PERIOD):
            batch_X = X_random[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
            batch_Y = Y_random[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
            optimizer.run(feed_dict = {input_images: batch_X, target_labels:batch_Y})

        print("Training epoch " + str(e+1))
        print("Accuracy: " + str(accuracy.eval(feed_dict={input_images: x_test, target_labels: y_test})))

    # After training, check for misclassified images during testing
    predictions = tf.argmax(digit_weights, 1)
    predicted_labels = predictions.eval(feed_dict={input_images: x_test})
    true_labels = np.argmax(y_test, axis=1)
    misclassified_indices = np.where(predicted_labels != true_labels)[0]
    print(f"Number of misclassified images: {len(misclassified_indices)}")

    # Count how many times each true label was misclassified
    misclassification_count = np.zeros(10, dtype=int)

    for index in misclassified_indices:
        true_label = true_labels[index]
        misclassification_count[true_label] += 1

    # Print the misclassification table
    print("\nMisclassification counts by digit (true label):")
    print(f"{'Digit':<10} {'Misclassified Count':<20}")
    print("-" * 30)
    for digit, count in enumerate(misclassification_count):
        print(f"{digit:<10} {count:<20}")


def classify_digits_v2():
    # Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Function to display a sample image and label
    def display_sample(num):
        label = np.argmax(y_train[num])  # Use NumPy's argmax to get the label
        image = x_train[num].reshape([28, 28])
        plt.title('Sample: %d  Label: %d' % (num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()

    display_sample(1234)

    # Display concatenated training images
    images = x_train[:500].reshape(-1, 28)
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

    # Build the neural network model using Keras API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    EPOCHS = 20
    BATCH_SIZE = 100

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')


def build_model(hidden_layers, hidden_nodes, learning_rate):
    input_images = tf.placeholder(tf.float32, shape=[None, 784])
    target_labels = tf.placeholder(tf.float32, shape=[None, 10])

    # Skip configurations where the length of hidden_nodes does not match hidden_layers
    if len(hidden_nodes) != hidden_layers:
        print(f"Skipping configuration: hidden_layers={hidden_layers}, hidden_nodes={hidden_nodes}")
        return None

    # Layer construction
    weights = []
    biases = []

    prev_layer_size = 784
    prev_layer = input_images

    for i in range(hidden_layers):
        weights.append(tf.Variable(tf.truncated_normal([prev_layer_size, hidden_nodes[i]])))
        biases.append(tf.Variable(tf.zeros([hidden_nodes[i]])))
        prev_layer = tf.nn.relu(tf.matmul(prev_layer, weights[-1]) + biases[-1])
        prev_layer_size = hidden_nodes[i]

    # Output layer
    output_weights = tf.Variable(tf.truncated_normal([prev_layer_size, 10]))
    output_biases = tf.Variable(tf.zeros([10]))
    digit_weights = tf.matmul(prev_layer, output_weights) + output_biases

    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=target_labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return input_images, target_labels, optimizer, accuracy


def train_and_evaluate_model(x_train, y_train, x_test, y_test, hidden_layers, hidden_nodes, learning_rate, batch_size):
    model_data = build_model(hidden_layers, hidden_nodes, learning_rate)
    
    if model_data is None:
        # Skip if build_model returned None due to mismatched hidden_layers and hidden_nodes
        return None

    input_images, target_labels, optimizer, accuracy = model_data
    
    # Session and training setup
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        EPOCHS = 20
        TRAIN_DATASIZE = x_train.shape[0]
        PERIOD = TRAIN_DATASIZE // batch_size

        for epoch in range(EPOCHS):
            idxs = np.random.permutation(TRAIN_DATASIZE)
            X_random = x_train[idxs]
            Y_random = y_train[idxs]

            for i in range(PERIOD):
                batch_X = X_random[i * batch_size:(i + 1) * batch_size]
                batch_Y = Y_random[i * batch_size:(i + 1) * batch_size]
                sess.run(optimizer, feed_dict={input_images: batch_X, target_labels: batch_Y})

        test_accuracy = sess.run(accuracy, feed_dict={input_images: x_test, target_labels: y_test})
        print(f"Test accuracy with hidden_layers={hidden_layers}, hidden_nodes={hidden_nodes}, learning_rate={learning_rate}, batch_size={batch_size}: {test_accuracy}")
        return test_accuracy


def grid_search(x_train, y_train, x_test, y_test):
    best_accuracy = 0.0
    best_params = None

    hidden_layer_options = [1, 2]  # Options for number of hidden layers
    hidden_node_options = [[512], [256, 128]]  # Options for hidden node sizes corresponding to the number of hidden layers
    learning_rate_options = [0.01, 0.1, 0.5, 1.0]  # Learning rate options
    batch_size_options = [50, 100, 200]  # Batch size options

    for hidden_layers in hidden_layer_options:
        for hidden_nodes in hidden_node_options:
            for learning_rate in learning_rate_options:
                for batch_size in batch_size_options:
                    # Train and evaluate the model with current configuration
                    accuracy = train_and_evaluate_model(x_train, y_train, x_test, y_test, hidden_layers, hidden_nodes, learning_rate, batch_size)

                    # Skip if the configuration was invalid and returned None
                    if accuracy is None:
                        continue

                    # Check if this is the best accuracy so far
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (hidden_layers, hidden_nodes, learning_rate, batch_size)

    print(f"Best accuracy: {best_accuracy} with parameters: hidden_layers={best_params[0]}, hidden_nodes={best_params[1]}, learning_rate={best_params[2]}, batch_size={best_params[3]}")


if __name__ == "__main__":
    if TF_VERSION == 1:
        import tensorflow.compat.v1 as tf
        from sklearn.model_selection import ParameterGrid
        tf.disable_v2_behavior()
        sess = tf.InteractiveSession()

        # Load MNIST data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        grid_search(x_train, y_train, x_test, y_test)
        #classify_digits_v1()
    elif TF_VERSION == 2:
        import tensorflow as tf
        classify_digits_v2()
    else:
        print("Error: Unexpected TensorFlow version.")