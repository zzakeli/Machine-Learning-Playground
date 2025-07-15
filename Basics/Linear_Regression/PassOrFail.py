import pandas as pd
# import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_excel("Basics/Linear_Regression/datasets/StudentPassOrFail.xlsx")

X = df.drop('Passed', axis=1).values
y = df['Passed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(y_test)
print(y_pred)


