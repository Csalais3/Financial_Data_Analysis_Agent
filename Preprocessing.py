import pandas as pd
import numpy as np

# Streamlining the data cleaning process
class DataCleaning:
    def __init__(self):
        pass
    
    def z_score(self, data):
        """
        Calculates the Z-scores for each feature in the dataset

        Args:
            data (np.array or pd.DataFrame): The input data, shaped as (samples, features)

        Returns:
            z_scores (np.array): The Z-scores for each data point, shaped as (samples, features)
        """
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number]).values  # Filter numeric columns
        return (data - np.mean(data, axis = 0)) / np.std(data, axis= 0)
    
    def remove_outliers(self, data, threshold= 3):
        """
        Removes outliers from the dataset using Z-scores

        Args:
            data (np.array): The input data, shaped as (samples, features)
            threshold (float): The Z-score threshold for identifying outliers

        Returns:
            clean_data (np.array): The data with outliers removed
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])  # Filter numeric columns
            z_scores = self.z_score(numeric_data)
            outlier_mask = (np.abs(z_scores) > threshold).any(axis=1)
            clean_data = data[~outlier_mask]  # Preserve DataFrame structure
        else:
            z_scores = self.z_score(data)
            outlier_mask = (np.abs(z_scores) > threshold).any(axis=1)
            clean_data = data[~outlier_mask]  # For NumPy arrays
            
        return clean_data
    
    def clean_data(self, data):
        """
        Cleans the data by handling missing values and removing outliers

        Args:
            data (pd.DataFrame): The input data, shaped as (samples, features)

        Returns:
            clean_data (pd.DataFrame): The cleaned data
        """
        print("Missing values before cleaning:")
        print(data.isnull().sum())  # Inspect missing values
        data = data.ffill() # Forward-fill missing values
        
        # Remove outliers
        clean_data = self.remove_outliers(data)
        
        return clean_data
    
# Implementation of MinMax Scaling for practice
class MinMaxScaling:
    def __init__(self, feature_range= (0, 1)):
        """
        Initializes the parameters (we have our feature range set to 0-1 so that its normalized)

        Args:
            feature_range (tuple): The desired range of transformed data, default is (0, 1)
        """
        self.feature_range = feature_range
        self.min = None
        self.max = None
    
    def fit(self, X):
        """
        calculates the minimum and maximum values of the data for scaling

        Args:
            X (np.array): The input data, shaped as (samples, features)
        """
        self.min = np.min(X, axis= 0)
        self.max = np.max(X, axis= 0)
    
    def transform(self, X):
        """
        Scales the data based on the computed min and max

        Args:
            X (np.array): The input data, shaped as (samples, features)

        Returns:
            X_Scaled (np.array): The scaled data, shaped as (samples, features)

        Raises:
            ValueError: If the scaler has not been fitted yet
        """
        # Check to see if our feature scalers have been fitted
        if self.min is None or self.max is None:
            raise ValueError("The scaler has not been fitted yet.")
        
        # Use the Min-Max Scaling formula found on the scikit website (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
        denominator = self.max - self.min 
        X_Scaled = np.where(denominator != 0, (X - self.min) / denominator, 1) # Handle division by zero using np.where
        
        #Scale to our feature range (in our case [0, 1])
        feature_min, feature_max = self.feature_range
        X_Scaled = X_Scaled * (feature_max - feature_min) + feature_min
        
        return X_Scaled
    
    def inverse_transform(self, X_Scaled):
        """
        Reverses the transformation done on the data

        Args:
            X_scaled (np.array): The scaled data, shaped as (samples, features)

        Returns:
            X (np.array): The original data, shaped as (samples, features)

        Raises:
            ValueError: If the scaler has not been fitted yet
        """
        # Check to see if our feature scalers have been fitted and prevent division by 0
        if self.min is None or self.max is None:
            raise ValueError("The scaler has not been fitted yet.")

        # Use the inverse Min-Max Scaling formula
        feature_min, feature_max = self.feature_range
        denominator = feature_max - feature_min

        # Handle division by zero using np.where
        X = np.where(denominator != 0, (X_Scaled - feature_min) / denominator, 1)
        X = X * (self.max - self.min) + self.min
        
        return X
    
    def fit_transform(self, X):
        """
        Fits the data, then transforms it without having to do both manually

        Args:
            X (np.array): The input data, shaped as (samples, features)

        Returns:
            X_Scaled np.array: The scaled data, shaped as (samples, features)
        """
        self.fit(X)
        return self.transform(X)

# Function for sequencing our time series 
def data_sequence(data, sequence_length):
    """
    Creates sequences of a specified (sequence_length) from the input data

    Args:
        data (np.array): The input data, shaped as (samples, features)
        sequence_length (int): The length of each sequence

    Returns:
        X (np.array): The sequences, shaped as (samples - sequence_length, sequence_length, features)
        y (np.array): The target values, shaped as (samples - sequence_length, features)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)

# Class setup
cleaner = DataCleaning()
scaler = MinMaxScaling(feature_range= (0, 1))

# Extracting the timeseries data from the csv file
data = pd.read_csv("/Users/csalais3/Downloads/Basic_Market_Prediction/Data/SPX.csv") # Our data (pd.DataFrame)
data_values = data.to_numpy() # Our data converted to a numpy array
cleaned_data = cleaner.clean_data(data) # Cleans our data 
dates = cleaned_data[['Date']] # Extracting our dates from the cleaned data

# Feature Engineering
cleaned_data['Returns'] = cleaned_data['Close'].pct_change()
cleaned_data['MA_10'] = cleaned_data['Close'].rolling(window=10).mean()
cleaned_data['Volatility'] = cleaned_data['Returns'].rolling(window=10).std()
cleaned_data = cleaned_data.dropna()  # Drop rows with NaN values

features = cleaned_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Returns', 'MA_10', 'Volatility']] # Extracting our numerical-valued features from the cleaned data

# Scaling our data 
scaled_data = scaler.fit_transform(features)

# Creating our data sequences
X, y = data_sequence(scaled_data, sequence_length= 30) # Monthly time sequencing

# Splitting the data into training and testing data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Confirms data preprocessing has run
print("Preprocessing complete! :)")
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


# After preprocessing, combine X_train, X_test, y_train, y_test back into DataFrames

# Flatten X_train and X_test
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten to 2D
X_test_flat = X_test.reshape(X_test.shape[0], -1)    # Flatten to 2D

# Extract the target column (e.g., 'Close')
target_column_index = 3 
y_train_target = y_train[:, target_column_index]
y_test_target = y_test[:, target_column_index]

# Create DataFrames
train_data = pd.DataFrame(X_train_flat)
train_data['Target'] = y_train_target

test_data = pd.DataFrame(X_test_flat)
test_data['Target'] = y_test_target

# Save to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Preprocessed data saved to train_data.csv and test_data.csv!")