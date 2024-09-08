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