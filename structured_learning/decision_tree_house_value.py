
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'House Size (sq ft)': [1500, 2000, 1750, 1450, 2100, 2200, 1900, 1600, 1850, 1700],
    'Age (years)': [10, 5, 7, 20, 3, 12, 6, 15, 8, 10],
    'Location': ['Suburb', 'City', 'Suburb', 'Rural', 'City', 'Suburb', 'Rural', 'City', 'Suburb', 'Rural'],
    'Price Category': ['High', 'High', 'High', 'Low', 'High', 'High', 'Low', 'Low', 'High', 'Low']  # Target variable
}
df = pd.DataFrame(data)

# Converting categorical data to numerical (One Hot Encoding)
df = pd.get_dummies(df, columns=['Location'])

# Preparing the feature matrix (X) and the target vector (y)
X = df.drop('Price Category', axis=1)
y = df['Price Category']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Calculating and printing the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example: Predicting with new data
# 1800 sq ft, 9 year, suburb = 0, rural = 1, city = 0
new_data = [[1800, 9, 0, 1, 0]]  # New data for prediction 
# Creating a DataFrame for new data with appropriate column names
new_data_df = pd.DataFrame(new_data, columns=X_train.columns)

# Making predictions with the new data DataFrame
new_prediction = model.predict(new_data_df)
print("Predicted Price Category:", new_prediction[0])