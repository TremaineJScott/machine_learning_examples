import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Sample data
data = {
    'Sweetness': [8, 3, 1, 7, 2, 2, 9, 7],
    'Crunchiness': [4, 6, 3, 5, 7, 8, 4, 6],
    'Color': ['Red', 'Yellow', 'Orange', 'Red', 'Yellow', 'Green', 'Red', 'Green'],
    'Fruit': ['Apple', 'Banana', 'Orange', 'Apple', 'Banana', 'Banana', 'Apple', 'Orange']
}
df = pd.DataFrame(data)

# Convert categorical data to numerical (One Hot Encoding)
df = pd.get_dummies(df, columns=['Color'])

# Feature scaling
scaler = StandardScaler()
features = ['Sweetness', 'Crunchiness']  # Only scale numerical features
df[features] = scaler.fit_transform(df[features])

# Preparing the feature matrix (X) and the target vector (y)
X = df.drop('Fruit', axis=1)
y = df['Fruit']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = svm.SVC(kernel='linear')  # Linear kernel
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Calculating and printing the accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example: Predicting with new data
new_data = {'Sweetness': 5, 'Crunchiness': 7, 'Color': 'Yellow'}
new_data_df = pd.DataFrame([new_data])

# Apply one-hot encoding to new data
new_data_df = pd.get_dummies(new_data_df)

# Ensure new data has the same features as the training data
new_data_df = new_data_df.reindex(columns=X_train.columns, fill_value=0)

# Scale the new data (only numerical features)
new_data_scaled = new_data_df.copy()
new_data_scaled[features] = scaler.transform(new_data_df[features])

# Making the prediction
new_prediction = model.predict(new_data_scaled)
print("Predicted Fruit:", new_prediction[0])

