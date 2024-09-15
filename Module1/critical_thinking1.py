from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to prompt user for number of variables and coefficients
def get_coefficients():
    while True:
        num_vars = int(input("Enter the number of variables (between 4 and 8): "))
        if 4 <= num_vars <= 8:
            break
        print("Please enter a valid number between 4 and 8.")
    
    coefficients = []
    for i in range(num_vars):
        coeff = float(input(f"Enter the coefficient for variable {i+1}: "))
        coefficients.append(coeff)
    
    return coefficients, num_vars

# Function to generate the dataset based on user coefficients
def generate_dataset(coefficients, num_vars, train_set_count, train_set_limit):
    train_input = []
    train_output = []

    for _ in range(train_set_count):
        vars = [randint(0, train_set_limit) for _ in range(num_vars)]
        result = sum(c * v for c, v in zip(coefficients, vars))
        train_input.append(vars)
        train_output.append(result)

    return train_input, train_output

# Prompt the user for coefficients and number of variables
coefficients, num_vars = get_coefficients()

# Constants
TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

# Generate training data
TRAIN_INPUT, TRAIN_OUTPUT = generate_dataset(coefficients, num_vars, TRAIN_SET_COUNT, TRAIN_SET_LIMIT)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(TRAIN_INPUT, TRAIN_OUTPUT, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prompt the user for input values to test the model
test_input = []
for i in range(num_vars):
    val = float(input(f"Enter the value for variable {i+1}: "))
    test_input.append(val)

# Reshape the input to a 2D array and make the prediction
predicted_output = model.predict([test_input])[0]

# Compute the actual output based on user coefficients and inputs
actual_output = sum(c * v for c, v in zip(coefficients, test_input))

# Print the predicted vs actual output
print(f'\nPredicted output for input {test_input}: {predicted_output}')
print(f'Actual output for input {test_input} based on the equation: {actual_output}')
