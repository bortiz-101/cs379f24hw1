import numpy as np
import pandas as pd
import os
dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, 'data')

# Load dataset
data = pd.read_csv(datadir + "/train.csv")

# Adaline class for weight computation
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

#Preprocessing data (guessing which is important maybe change later ***)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'].fillna(data['Age'].mean(), inplace=True)

#Set typical variables for features and outcomes
#Features
X = data[['Pclass', 'Sex', 'Age', 'Fare']].values
#Outcome
y = data['Survived'].values