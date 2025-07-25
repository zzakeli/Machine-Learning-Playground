List of 20% that needs to be remembered in order to learn ML effectively.
This includes models, metrics, preprocessing, and pipelines.

Terms
-Target variable
-feature
-model
-metrics
-hyperparameter
-preprocessing
-pipeline

Types of Supervised Learning
-Classification
-Regression

Classification
Groupings of each side based on the features/data, like dividing them based on the which side it should be.
The model learns and classify which side this data belongs to.

Regression
Predict a continuous value such us a salary, glucose levels, rent value, etc.

Models I used (for now) for Classification
-KNN
-Logistic Regression

Metrics I used for Classification
-accuracy_score
-confusion_matrix
-classification_report
-roc_curve
-auc

Model Selection I used for Classification
-train_test_split (X,y,test_size, random_state) #known for splitting the data to train and test set
-cross_val_score (model, X, y, cv, scoring) #'s scoring='accuracy' #known as cross validation
-GridSearchCV (model, params, cv) #finding which hyperparameters are best to use
-KFold (folds, shuffle, random_state) #an input number of cv for cross_val_score or GridSearchCV

Models I used (for now) for Regression
-Linear Regression
-Ridge
-Lasso

Model Selection I used for Regression
-train_test_split (X,y,test_size, random_state) #known for splitting the data to train and test set
-cross_val_score (model, X, y, cv, scoring) #'s scoring='r2' #known as cross validation
-GridSearchCV (model, params, cv) #finding which hyperparameters are best to use
-KFold (folds, shuffle, random_state) #an input number of cv for cross_val_score or GridSearchCV

Metrics I used for Regression
-mean_squared_error
-rmse
