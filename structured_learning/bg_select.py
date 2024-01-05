import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Assuming your file is named 'bgg_db_1806_50.csv' and is located in the same directory as your script
file_path = 'structured_learning/mnt/data/bgg_db_1806_50.csv'
board_games_data = pd.read_csv(file_path)
# Drop the column with the game names
board_games_data = board_games_data.drop('names', axis=1)

# Display the first few rows of the dataframe
print(board_games_data.head())
# Check for missing values
print(board_games_data.isnull().sum())

# One-hot encoding for categorical text data
mechanic_dummies = board_games_data['mechanic'].str.get_dummies(sep=', ')
category_dummies = board_games_data['category'].str.get_dummies(sep=', ')

# Combine the one-hot encoded columns with the original dataframe
processed_data = pd.concat([board_games_data, mechanic_dummies, category_dummies], axis=1)

# Drop the original text columns
processed_data.drop(['mechanic', 'category'], axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler

# Extracting numeric columns for scaling
numeric_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('my_rating')

# Standardizing the numeric features
scaler = StandardScaler()
processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])


# Preparing data for training
X = processed_data.drop('my_rating', axis=1)
y = processed_data['my_rating']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting ratings
y_pred = model.predict(X_test)

# Calculating MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example new game data
new_game_data = {
    'max_time': 30,  # Example value
    'mechanic': 'Area Control / Area Influence, Tile Placement',  # Example value
    'category': 'Puzzle, Real-time',  # Example value
    'weight': 2.3 # Example value
}

# Convert to DataFrame
new_game_df = pd.DataFrame([new_game_data])

# One-hot encode 'mechanic' and 'category' (use the same columns as in your training data)
new_game_mechanic_dummies = new_game_df['mechanic'].str.get_dummies(sep=', ')
new_game_category_dummies = new_game_df['category'].str.get_dummies(sep=', ')

# Combine with the new game data
new_game_processed = pd.concat([new_game_df, new_game_mechanic_dummies, new_game_category_dummies], axis=1)

# Drop the original text columns
new_game_processed.drop(['mechanic', 'category'], axis=1, inplace=True)

# Ensure all columns from the training data are present, filling missing ones with 0
for col in X.columns:
    if col not in new_game_processed.columns:
        new_game_processed[col] = 0

# Reorder columns to match training data
new_game_processed = new_game_processed[X.columns]

# Apply the same scaling to the numeric features
new_game_processed[numeric_columns] = scaler.transform(new_game_processed[numeric_columns])

# Predict
predicted_rating = model.predict(new_game_processed)
print(f"Predicted Rating: {predicted_rating[0]}")