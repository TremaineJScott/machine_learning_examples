import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data
file_path = 'C:\\Users\\ScottTremaine\\source\\repos\\MachineLearning\\structured_learning\\house_prices_dataset.csv'  
data = pd.read_csv(file_path)

# One Hot Encoding for the 'Location' column
# We use ColumnTransformer to apply this transformation
ct = ColumnTransformer([
    ('one_hot_encoder', OneHotEncoder(), ['Location'])
], remainder='passthrough')  # 'passthrough' means other columns are not affected

# Preparing the feature matrix (X) and the target vector (y)
X = ct.fit_transform(data[['House Size (sq ft)', 'Age (years)', 'Location']])
y = data['House Price ($)']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Calculating and printing the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predicting with new data
# Ensure the format of new data matches the training data
new_data = [[2000, 15, 'Suburb']]  # New data for prediction
# Convert new data to DataFrame with appropriate column names
new_data_df = pd.DataFrame(new_data, columns=['House Size (sq ft)', 'Age (years)', 'Location'])

# Now transform the new data using the ColumnTransformer
new_data_transformed = ct.transform(new_data_df)

# Making predictions with the transformed data
new_prediction = model.predict(new_data_transformed)
print("Predicted House Price:", new_prediction[0])
