import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

dataset = pd.read_csv("breast-cancer.csv")
# dataset = datasets.load_breast_cancer()
# X, y = dataset.data, dataset.target
train, test = train_test_split(dataset, test_size=0.2)

y_train = np.array([0 if y == "B" else 1 for y in train["diagnosis"]])
train = train.drop(["id", "diagnosis"], axis="columns")
X_train = np.array(train.values.tolist())

model = LogisticRegression(alpha=0.00001)
model.gradient_descent(X_train, y_train)
y_pred = model.predict(X_train)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test) / len(y_test)

acc = accuracy(y_pred, y_train)
print("The model's accuracy is = ", acc)