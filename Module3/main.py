import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Generate the training data
def generate_nums():
    np.random.seed(101)
    tf.random.set_seed(101)  # For reproducibility in TensorFlow 2.x

    # Generate random linear data
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)

    # Adding noise to the random linear data
    x += np.random.uniform(-4, 4, 50)
    y += np.random.uniform(-4, 4, 50)
    n = len(x)  # Number of data points

    return x, y, n

# Step 2: Define the model and placeholders (simulated in eager execution)
x_train, y_train, n = generate_nums()
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Hyperparameters
learning_rate = 0.0001
training_epochs = 1000

# Step 3: Initialize trainable variables (weights and bias)
W = tf.Variable(np.random.normal(), dtype=tf.float32, name="weight")
b = tf.Variable(np.random.normal(), dtype=tf.float32, name="bias")

# Step 4: Hypothesis (prediction) function: y = W * x + b
def model(x):
    return W * x + b

# Step 5: Cost function (mean squared error)
def cost(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Step 6: Optimizer (Gradient Descent)
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

# Step 7: Training function
def train_step():
    with tf.GradientTape() as tape:
        predictions = model(x_train)
        loss = cost(predictions, y_train)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    return loss

# Step 8: Training loop
for epoch in range(training_epochs):
    loss = train_step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{training_epochs}, Loss: {loss.numpy()}, Weight: {W.numpy()}, Bias: {b.numpy()}')

# Step 9: Print the final results
print(f"Final training cost: {loss.numpy()}")
print(f"Final weight: {W.numpy()}")
print(f"Final bias: {b.numpy()}")

# Step 10: Plot the original data and the fitted line
plt.figure(figsize=(8, 6))

# Plot the original training data points
plt.scatter(x_train, y_train, color='blue', label='Training data')

# Plot the fitted line (predictions from the model)
y_pred = model(x_train)  # Get the predictions from the trained model
plt.plot(x_train, y_pred, color='red', label='Fitted line')

# Add labels and legend
plt.title('Fitted Line on Top of Training Data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
