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

# Min-Max Normalization function
def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

# Apply Min-Max Normalization
X_train = min_max_scaling(X_train)
X_test = min_max_scaling(X_test)

# Initialize Adaline
adaline = AdalineGD(eta=0.001, n_iter=500)
adaline.fit(X_train, y_train)

#Adaline prediction
predictions = adaline.predict(X_test)

#Save predictions to a CSV file
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
output.to_csv('predictions.csv', index=False)

#Initialize variable for predictions.csv
predictions_data = pd.read_csv('predictions.csv')

# Merge the dataframes on 'PassengerId'
merged_data = pd.merge(test, predictions_data, on='PassengerId')

# Save the merged dataframe to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)

# DataFrame creation
data = {
    'Pclass': [1, 2, 3, 1, 2, 3],
    'Gender': ['male', 'female', 'female', 'male', 'female', 'male'],
    'Age': [22, 35, 58, 24, 30, 45],
    'Fare': [50, 20, 10, 60, 25, 15],
    'Survived': [1, 0, 1, 1, 0, 0]
}
merged_data = pd.DataFrame(data)

# Calculate survival rates
print(f"Overall Survival Rate: {merged_data['Survived'].mean():.2f}")

print("\nSurvival Rate by Pclass:")
print(merged_data.groupby('Pclass')['Survived'].mean())

print("\nSurvival Rate by Gender:")
print(merged_data.groupby('Gender')['Survived'].mean())

# Create bins and calculate survival rates
bins = {'Age': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'Fare': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

labels = {'Age': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
          'Fare': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']}

for column in ['Age', 'Fare']:
    merged_data[f'{column}Group'] = pd.cut(merged_data[column], bins=bins[column], labels=labels[column])
    print(f"\nSurvival Rate by {column} Group:")
    print(merged_data.groupby(f'{column}Group', observed=True)['Survived'].mean())