import numpy as np
import pandas as pd
import os
dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, 'data')

# Load dataset
train = pd.read_csv(datadir + "/train.csv")
test = pd.read_csv(datadir + "/test.csv")

# Adaline class
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, net_input):
        return 1 / (1 + np.exp(-net_input))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

#Preprocessing training data (guessing which is important maybe change later ***)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

#Preprocessing test data (guessing which is important maybe change later ***)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

#Set typical variables for features and outcomes
#Features
X_train = train[['Pclass', 'Sex', 'Age', 'Fare']].values
X_test = test[['Pclass', 'Sex', 'Age', 'Fare']].values

#Outcome
y_train = train['Survived'].values

# Standardization
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Apply standardization to your training and test sets
X_train_standardized = standardize(X_train)
X_test_standardized = standardize(X_test)

#Adaline prediction
adaline = AdalineGD(eta=0.001, n_iter=100)
adaline.fit(X_train_standardized, y_train)
predictions = adaline.predict(X_test_standardized)

#Save predictions to a CSV file
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
output.to_csv('predictions.csv', index=False)

#Initialize variable for predictions.csv
predictions_data = pd.read_csv('predictions.csv')

# Merge the dataframes on 'PassengerId'
merged_data = pd.merge(test, predictions_data, on='PassengerId')

# Save the merged dataframe to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)

# Read merged data
merged_data = pd.read_csv('merged_data.csv')

# Calculate survival rates
print(f"Overall Survival Rate: {merged_data['Survived'].mean():.2f}")

print("\nSurvival Rate by Pclass:")
print(merged_data.groupby('Pclass', observed=True)['Survived'].mean())

print("\nSurvival Rate by Sex:")
print(merged_data.groupby('Sex', observed=True)['Survived'].mean())

# Bins and labels for Age and Fare
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

fare_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
fare_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-200', '200-300']

# Calculate survival rates for Age
merged_data['AgeGroup'] = pd.cut(merged_data['Age'], bins=age_bins, labels=age_labels)
print("\nSurvival Rate by Age Group:")
print(merged_data.groupby('AgeGroup', observed=True)['Survived'].mean())

# Calculate survival rates for Fare
merged_data['FareGroup'] = pd.cut(merged_data['Fare'], bins=fare_bins, labels=fare_labels)
print("\nSurvival Rate by Fare Group:")
print(merged_data.groupby('FareGroup', observed=True)['Survived'].mean())
print("Min Fare:", merged_data['Fare'].min())
print("Max Fare:", merged_data['Fare'].max())
print(merged_data['Fare'].describe())