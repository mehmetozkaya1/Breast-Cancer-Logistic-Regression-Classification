import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("breast-cancer.csv")

radius_mean = df["radius_mean"][:500]
texture_mean = df["texture_mean"][:500]

y_train = df["diagnosis"][:500]

x_train = np.column_stack((radius_mean, texture_mean))

x_train = np.array(x_train)
y_train = np.array(y_train)

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_train)

print(y_pred)

score = lr_model.score(x_train, y_train)

print(score)