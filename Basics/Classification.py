import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=42, 
                                                    stratify=y)

# neighborss = np.arange(1,70)
# train_accuracy = {}
# test_accuracy = {}
# for neighbors in neighborss:
#     model = KNeighborsClassifier(n_neighbors=neighbors)
#     model.fit(X_train, y_train)
#     #model.predict(X_test, y_test)
#     train_accuracy[neighbors] = model.score(X_train, y_train)
#     test_accuracy[neighbors] = model.score(X_test, y_test)

# plt.plot(neighborss, train_accuracy.values(), label="Training Accuracy")
# plt.plot(neighborss, test_accuracy.values(), label="Testing Accuracy")
# plt.title("Number of k Neighbors")

# print(test_accuracy)
# plt.show()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_test)
print()
print(y_pred)


 