import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
df = pd.read_csv("datasets/dataset-uci.csv")

print(df.head())
