import numpy as np
import pandas as pd
import os
dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, 'data')

# Load dataset
train = pd.read_csv(datadir + "/train.csv")
test = pd.read_csv(datadir + "/test.csv")

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

#Preprocessing training data (guessing which is important maybe change later ***)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Age'] = train['Age'].fillna(train['Age'].mean())

#Preprocessing test data (guessing which is important maybe change later ***)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Age'] = test['Age'].fillna(test['Age'].mean())

#Set typical variables for features and outcomes
#Features
X_train = train[['Pclass', 'Sex', 'Age', 'Fare']].values
X_test = test[['Pclass', 'Sex', 'Age', 'Fare']].values

#Outcome
y_train = train['Survived'].values


# Initialize Adaline
adaline = AdalineGD(eta=0.01, n_iter=50)
adaline.fit(X_train, y_train)

#Adaline prediction
predictions = adaline.predict(X_test)

#Save predictions to a CSV file
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
output.to_csv('predictions.csv', index=False)