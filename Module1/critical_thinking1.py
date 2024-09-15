from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Constants for generating the dataset
TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

# Generating training data
TRAIN_INPUT = []
TRAIN_OUTPUT = []

for _ in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (2 * b) + (3 * c)  # Algebraic equation
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(TRAIN_INPUT, TRAIN_OUTPUT, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Testing the model with a new input
test_input = [[10, 20, 30]]  # Example input
predicted_output = model.predict(test_input)
print(f'Predicted output for input {test_input}: {predicted_output[0]}')
