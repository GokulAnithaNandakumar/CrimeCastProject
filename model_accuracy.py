

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

# Load the datasets
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')


# In[19]:


# Extract date and time features
def extract_date_features(df):
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'])
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'])
    df['Year_Reported'] = df['Date_Reported'].dt.year
    df['Month_Reported'] = df['Date_Reported'].dt.month
    df['Day_Reported'] = df['Date_Reported'].dt.day
    df['Hour_Occurred'] = df['Date_Occurred'].dt.hour
    df['Day_Of_Week_Reported'] = df['Date_Reported'].dt.dayofweek
    df['Day_Of_Week_Occurred'] = df['Date_Occurred'].dt.dayofweek
    df.drop(columns=['Date_Reported', 'Date_Occurred'], inplace=True)

# Apply feature extraction
extract_date_features(train_data)
extract_date_features(test_data)

# Separate features and target variable from training data
X = train_data.drop(columns=['Crime_Category'])
y = train_data['Crime_Category']

# Identify numeric and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines for numeric and categorical data
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)


# In[20]:


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model: Logistic Regression with basic settings
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))  # Increased max_iter
])

baseline_pipeline.fit(X_train, y_train)
baseline_val_accuracy = accuracy_score(y_val, baseline_pipeline.predict(X_val))
print(f"Baseline Logistic Regression Validation Accuracy: {baseline_val_accuracy:.4f}")


# In[21]:


# # Save baseline predictions to CSV
# test_predictions_baseline = baseline_pipeline.predict(test_data)
# submission_df_baseline = pd.DataFrame({
#     'ID': range(1, len(test_predictions_baseline) + 1),
#     'Crime_Category': test_predictions_baseline
# })
# submission_df_baseline.to_csv('submission_baseline.csv', index=False)


# In[22]:


# Baseline model: Logistic Regression with basic settings
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))  # Increased max_iter
])

baseline_pipeline.fit(X_train, y_train)
baseline_val_accuracy = accuracy_score(y_val, baseline_pipeline.predict(X_val))
print(f"Baseline Logistic Regression Validation Accuracy: {baseline_val_accuracy:.4f}")


# In[23]:


# Polynomial Regression pipeline
poly_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))
])

# Fit and evaluate the pipeline
poly_pipeline.fit(X_train, y_train)
poly_val_accuracy = accuracy_score(y_val, poly_pipeline.predict(X_val))
print(f"Polynomial Regression Validation Accuracy: {poly_val_accuracy:.4f}")


# In[24]:


# Save polynomial regression predictions to CSV
test_predictions_poly = poly_pipeline.predict(test_data)
submission_df_poly = pd.DataFrame({
    'ID': range(1, len(test_predictions_poly) + 1),
    'Crime_Category': test_predictions_poly
})
submission_df_poly.to_csv('submission.csv', index=False)


# In[25]:


# Logistic Regression with cross-validation and hyperparameter tuning
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))  # Increased max_iter
])

param_grid_logistic = {
    'classifier__C': np.logspace(-4, 4, 20),
    'classifier__solver': ['liblinear', 'lbfgs']
}

# Define cross-validation strategy with reduced number of folds
cv_reduced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_logistic = RandomizedSearchCV(logistic_pipeline, param_distributions=param_grid_logistic, n_iter=10, cv=cv_reduced, verbose=1, random_state=42, n_jobs=-1)
random_search_logistic.fit(X_train, y_train)
logistic_val_accuracy = accuracy_score(y_val, random_search_logistic.predict(X_val))
print(f"Logistic Regression Validation Accuracy: {logistic_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_logistic.best_params_}")


# In[26]:


# # Save logistic regression predictions to CSV
# test_predictions_logistic = random_search_logistic.predict(test_data)
# submission_df_logistic = pd.DataFrame({
#     'ID': range(1, len(test_predictions_logistic) + 1),
#     'Crime_Category': test_predictions_logistic
# })
# submission_df_logistic.to_csv('submission_logistic.csv', index=False)


# In[ ]:





# In[27]:


# Decision Tree with cross-validation and hyperparameter tuning
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

param_grid_decision_tree = {
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Define cross-validation strategy with reduced number of folds
cv_reduced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_decision_tree = RandomizedSearchCV(decision_tree_pipeline, param_distributions=param_grid_decision_tree, n_iter=10, cv=cv_reduced, verbose=1, random_state=42, n_jobs=-1)
random_search_decision_tree.fit(X_train, y_train)
decision_tree_val_accuracy = accuracy_score(y_val, random_search_decision_tree.predict(X_val))
print(f"Decision Tree Validation Accuracy: {decision_tree_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_decision_tree.best_params_}")


# In[28]:


# # Predict on the test set and save to CSV
# test_predictions_decision_tree = random_search_decision_tree.predict(test_data)
# submission_df_decision_tree = pd.DataFrame({
#     'ID': range(1, len(test_predictions_decision_tree) + 1),
#     'Crime_Category': test_predictions_decision_tree
# })
# submission_df_decision_tree.to_csv('submission_decision_tree.csv', index=False)


# In[29]:


# K-Nearest Neighbors with cross-validation and hyperparameter tuning
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

# Define cross-validation strategy with reduced number of folds
cv_reduced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_knn = RandomizedSearchCV(knn_pipeline, param_distributions=param_grid_knn, n_iter=10, cv=cv_reduced, verbose=1, random_state=42, n_jobs=-1)
random_search_knn.fit(X_train, y_train)
knn_val_accuracy = accuracy_score(y_val, random_search_knn.predict(X_val))
print(f"K-Nearest Neighbors Validation Accuracy: {knn_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_knn.best_params_}")


# In[30]:


# # Predict on the test set and save to CSV
# test_predictions_knn = random_search_knn.predict(test_data)
# submission_df_knn = pd.DataFrame({
#     'ID': range(1, len(test_predictions_knn) + 1),
#     'Crime_Category': test_predictions_knn
# })
# submission_df_knn.to_csv('submission_knn.csv', index=False)


# In[31]:


# Support Vector Classifier with cross-validation and hyperparameter tuning
svc_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

param_grid_svc = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear']
}

# Define cross-validation strategy with reduced number of folds
cv_reduced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_svc = RandomizedSearchCV(svc_pipeline, param_distributions=param_grid_svc, n_iter=10, cv=cv_reduced, verbose=1, random_state=42, n_jobs=-1)
random_search_svc.fit(X_train, y_train)
svc_val_accuracy = accuracy_score(y_val, random_search_svc.predict(X_val))
print(f"Support Vector Classifier Validation Accuracy: {svc_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_svc.best_params_}")


# In[32]:


# # Predict on the test set and save to CSV
# test_predictions_svc = random_search_svc.predict(test_data)
# submission_df_svc = pd.DataFrame({
#     'ID': range(1, len(test_predictions_svc) + 1),
#     'Crime_Category': test_predictions_svc
# })
# submission_df_svc.to_csv('submission_svc.csv', index=False)


# In[16]:


# Random Forest with cross-validation and hyperparameter tuning
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid_random_forest = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Define cross-validation strategy with reduced number of folds
cv_reduced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_random_forest = RandomizedSearchCV(random_forest_pipeline, param_distributions=param_grid_random_forest, n_iter=10, cv=cv_reduced, verbose=1, random_state=42, n_jobs=-1)
random_search_random_forest.fit(X_train, y_train)
random_forest_val_accuracy = accuracy_score(y_val, random_search_random_forest.predict(X_val))
print(f"Random Forest Validation Accuracy: {random_forest_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_random_forest.best_params_}")


# In[34]:


# Bagging Classifier with cross-validation and hyperparameter tuning
bagging_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BaggingClassifier(random_state=42))
])

param_grid_bagging = {
    'classifier__n_estimators': [10, 50],
    'classifier__max_samples': [0.5, 1.0],
    'classifier__max_features': [0.5, 1.0]
}

random_search_bagging = RandomizedSearchCV(bagging_pipeline, param_distributions=param_grid_bagging, n_iter=3, cv=3, verbose=1, random_state=42, n_jobs=-1)
random_search_bagging.fit(X_train, y_train)
bagging_val_accuracy = accuracy_score(y_val, random_search_bagging.predict(X_val))
print(f"Bagging Classifier Validation Accuracy: {bagging_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_bagging.best_params_}")


# In[35]:


# Gradient Boosting with cross-validation and hyperparameter tuning
gradient_boosting_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

param_grid_gradient_boosting = {
    'classifier__n_estimators': [50],
    'classifier__learning_rate': [0.1],
    'classifier__max_depth': [3]
}

# Define cross-validation strategy with reduced number of folds
cv_reduced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_gradient_boosting = RandomizedSearchCV(gradient_boosting_pipeline, param_distributions=param_grid_gradient_boosting, n_iter=3, cv=cv_reduced, verbose=1, random_state=42, n_jobs=-1)
print("Starting Gradient Boosting training...")
random_search_gradient_boosting.fit(X_train, y_train)
print("Gradient Boosting training completed.")

gradient_boosting_val_accuracy = accuracy_score(y_val, random_search_gradient_boosting.predict(X_val))
print(f"Gradient Boosting Validation Accuracy: {gradient_boosting_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_gradient_boosting.best_params_}")


# In[36]:


# # Predict on the test set and save to CSV
# test_predictions_gradient_boosting = random_search_gradient_boosting.predict(test_data)
# submission_df_gradient_boosting = pd.DataFrame({
#     'ID': range(1, len(test_predictions_gradient_boosting) + 1),
#     'Crime_Category': test_predictions_gradient_boosting
# })
# submission_df_gradient_boosting.to_csv('submission_gradient_boosting.csv', index=False)


# In[37]:


# Multi-layer Perceptron (MLP) with cross-validation and hyperparameter tuning
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(max_iter=300))
])

param_grid_mlp = {
    'classifier__hidden_layer_sizes': [(50,), (100,)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__solver': ['adam'],
    'classifier__alpha': [0.0001]
}

random_search_mlp = RandomizedSearchCV(mlp_pipeline, param_distributions=param_grid_mlp, n_iter=3, cv=3, verbose=1, random_state=42, n_jobs=-1)
random_search_mlp.fit(X_train, y_train)
mlp_val_accuracy = accuracy_score(y_val, random_search_mlp.predict(X_val))
print(f"MLP Validation Accuracy: {mlp_val_accuracy:.4f}")
print(f"Best Parameters: {random_search_mlp.best_params_}")


# In[38]:


# # Save MLP predictions to CSV
# test_predictions_mlp = random_search_mlp.predict(test_data)
# submission_df_mlp = pd.DataFrame({
#     'ID': range(1, len(test_predictions_mlp) + 1),
#     'Crime_Category': test_predictions_mlp
# })
# submission_df_mlp.to_csv('submission_mlp.csv', index=False)


# 2nd BEST WORKING CODE BUT WITHOUT HYPERPARAMETER TUNING

# In[22]:


# # Predict on the test set and save to CSV
# test_predictions_bagging = random_search_bagging.predict(test_data)
# submission_df_bagging = pd.DataFrame({
#     'ID': range(1, len(test_predictions_bagging) + 1),
#     'Crime_Category': test_predictions_bagging
# })
# submission_df_bagging.to_csv('submission_bagging.csv', index=False)


# In[23]:


# # Function to extract date and time features
# def extract_date_features(df):
#     df['Date_Reported'] = pd.to_datetime(df['Date_Reported'])
#     df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'])
#     df['Year_Reported'] = df['Date_Reported'].dt.year
#     df['Month_Reported'] = df['Date_Reported'].dt.month
#     df['Day_Reported'] = df['Date_Reported'].dt.day
#     df['Hour_Occurred'] = df['Date_Occurred'].dt.hour
#     df['Day_Of_Week_Reported'] = df['Date_Reported'].dt.dayofweek
#     df['Day_Of_Week_Occurred'] = df['Date_Occurred'].dt.dayofweek
#     df.drop(columns=['Date_Reported', 'Date_Occurred'], inplace=True)

# # Apply feature extraction
# extract_date_features(train_data)
# extract_date_features(test_data)

# # Separate features and target variable from training data
# X = train_data.drop(columns=['Crime_Category'])
# y = train_data['Crime_Category']

# # Identify numeric and categorical features
# num_features = X.select_dtypes(include=['int64', 'float64']).columns
# cat_features = X.select_dtypes(include=['object']).columns

# # Preprocessing pipelines for numeric and categorical data
# num_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# cat_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])


# In[24]:


# # Combine preprocessing steps
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', num_transformer, num_features),
#         ('cat', cat_transformer, cat_features)
#     ]
# )

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# # Logistic Regression
# logistic_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', LogisticRegression(max_iter=1000))
# ])
# logistic_pipeline.fit(X_train, y_train)
# logistic_val_accuracy = accuracy_score(y_val, logistic_pipeline.predict(X_val))
# print(f"Logistic Regression Validation Accuracy: {logistic_val_accuracy:.4f}")


# In[26]:


# # Decision Tree
# decision_tree_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', DecisionTreeClassifier(random_state=42))
# ])
# decision_tree_pipeline.fit(X_train, y_train)
# decision_tree_val_accuracy = accuracy_score(y_val, decision_tree_pipeline.predict(X_val))
# print(f"Decision Tree Validation Accuracy: {decision_tree_val_accuracy:.4f}")


# In[27]:


# # K-Nearest Neighbors
# knn_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', KNeighborsClassifier())
# ])
# knn_pipeline.fit(X_train, y_train)
# knn_val_accuracy = accuracy_score(y_val, knn_pipeline.predict(X_val))
# print(f"K-Nearest Neighbors Validation Accuracy: {knn_val_accuracy:.4f}")


# In[28]:


# # Support Vector Classifier
# svc_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', SVC())
# ])
# svc_pipeline.fit(X_train, y_train)
# svc_val_accuracy = accuracy_score(y_val, svc_pipeline.predict(X_val))
# print(f"Support Vector Classifier Validation Accuracy: {svc_val_accuracy:.4f}")# Support Vector Classifier
# svc_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', SVC())
# ])
# svc_pipeline.fit(X_train, y_train)
# svc_val_accuracy = accuracy_score(y_val, svc_pipeline.predict(X_val))
# print(f"Support Vector Classifier Validation Accuracy: {svc_val_accuracy:.4f}")


# In[29]:


# # Random Forest
# random_forest_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])
# random_forest_pipeline.fit(X_train, y_train)
# random_forest_val_accuracy = accuracy_score(y_val, random_forest_pipeline.predict(X_val))
# print(f"Random Forest Validation Accuracy: {random_forest_val_accuracy:.4f}")


# In[30]:


# # Gradient Boosting
# gradient_boosting_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', GradientBoostingClassifier(random_state=42))
# ])
# gradient_boosting_pipeline.fit(X_train, y_train)
# gradient_boosting_val_accuracy = accuracy_score(y_val, gradient_boosting_pipeline.predict(X_val))
# print(f"Gradient Boosting Validation Accuracy: {gradient_boosting_val_accuracy:.4f}")


# In[31]:


# # Bagging Classifier
# bagging_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', BaggingClassifier(random_state=42))
# ])
# bagging_pipeline.fit(X_train, y_train)
# bagging_val_accuracy = accuracy_score(y_val, bagging_pipeline.predict(X_val))
# print(f"Bagging Classifier Validation Accuracy: {bagging_val_accuracy:.4f}")


# In[32]:


# # Select the best model for final prediction
# best_model_pipeline = gradient_boosting_pipeline  # Replace with the best model based on validation accuracy

# # Fit the best model on the entire training data
# best_model_pipeline.fit(X_train, y_train)

# # Predict on the test set
# test_predictions = best_model_pipeline.predict(test_data)

# # Create submission DataFrame
# submission_df = pd.DataFrame({
#     'ID': range(1, len(test_predictions) + 1),
#     'Crime_Category': test_predictions
# })

# # Save submission to CSV
# submission_df.to_csv('submission.csv', index=False)


# OLD WORKING CODE

# In[33]:


# # Check for missing values
# print(train_df.isnull().sum())

# # Data types and summary statistics
# print(train_df.info())
# print(train_df.describe())

# # Dropping non-numeric columns for correlation matrix
# numeric_df = train_df.select_dtypes(include=[np.number])

# # Correlation matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(numeric_df.corr(), annot=True)
# plt.show()


# In[34]:


# # Extracting date and time features
# train_df['Date_Reported'] = pd.to_datetime(train_df['Date_Reported'])
# train_df['Date_Occurred'] = pd.to_datetime(train_df['Date_Occurred'])

# # Example of feature extraction
# train_df['Year_Reported'] = train_df['Date_Reported'].dt.year
# train_df['Month_Reported'] = train_df['Date_Reported'].dt.month
# train_df['Day_Reported'] = train_df['Date_Reported'].dt.day
# train_df['Hour_Occurred'] = train_df['Date_Occurred'].dt.hour
# train_df['Day_Of_Week_Reported'] = train_df['Date_Reported'].dt.dayofweek
# train_df['Day_Of_Week_Occurred'] = train_df['Date_Occurred'].dt.dayofweek

# # Dropping the original Date columns
# train_df = train_df.drop(columns=['Date_Reported', 'Date_Occurred'])

# # Apply the same transformations to the test data
# test_df['Date_Reported'] = pd.to_datetime(test_df['Date_Reported'])
# test_df['Date_Occurred'] = pd.to_datetime(test_df['Date_Occurred'])

# test_df['Year_Reported'] = test_df['Date_Reported'].dt.year
# test_df['Month_Reported'] = test_df['Date_Reported'].dt.month
# test_df['Day_Reported'] = test_df['Date_Reported'].dt.day
# test_df['Hour_Occurred'] = test_df['Date_Occurred'].dt.hour
# test_df['Day_Of_Week_Reported'] = test_df['Date_Reported'].dt.dayofweek
# test_df['Day_Of_Week_Occurred'] = test_df['Date_Occurred'].dt.dayofweek

# test_df = test_df.drop(columns=['Date_Reported', 'Date_Occurred'])


# In[35]:


# # Preprocessing pipeline
# numerical_features = ['Latitude', 'Longitude', 'Time_Occurred', 'Victim_Age', 'Year_Reported', 'Month_Reported', 'Day_Reported', 'Hour_Occurred', 'Day_Of_Week_Reported', 'Day_Of_Week_Occurred']
# categorical_features = ['Location', 'Cross_Street', 'Area_Name', 'Victim_Sex', 'Victim_Descent', 'Premise_Description', 'Weapon_Description']

# numerical_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])


# In[36]:


# # Separate features and target variable
# X = train_df.drop(columns=['Crime_Category'])
# y = train_df['Crime_Category']

# # Applying the preprocessing steps to the data
# X = preprocessor.fit_transform(X)
# X_test = preprocessor.transform(test_df)


# In[37]:


# # Splitting the data into training and validation sets
# # change test size
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=40)


# In[38]:


# # Logistic Regression
# logistic_model = LogisticRegression(max_iter=1000)
# logistic_model.fit(X_train, y_train)

# # Predicting on the validation set
# y_pred_logistic = logistic_model.predict(X_val)
# logistic_accuracy = accuracy_score(y_val, y_pred_logistic)
# print(f'Logistic Regression Accuracy: {logistic_accuracy:.4f}')

# # Predicting on the test set
# test_predictions_logistic = logistic_model.predict(X_test)

# # # Save predictions to CSV file
# # submission_logistic = pd.DataFrame({'ID': test_df.index + 1, 'Crime_Category': test_predictions_logistic})
# # submission_logistic.to_csv('submission_Logistic_Regression.csv', index=False)


# In[39]:


# # Random Forest
# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X_train, y_train)

# # Predicting on the validation set
# y_pred_rf = rf.predict(X_val)
# rf_accuracy = accuracy_score(y_val, y_pred_rf)
# print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

# # Predicting on the test set
# test_predictions_rf = rf.predict(X_test)

# # # Save predictions to CSV file
# # submission_rf = pd.DataFrame({'ID': test_df.index + 1, 'Crime_Category': test_predictions_rf})
# # submission_rf.to_csv('submission_Random_Forest.csv', index=False)


# In[ ]:





# In[40]:


# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# import xgboost as xgb
# import warnings
# warnings.filterwarnings("ignore")

# # Load the datasets
# train_df = pd.read_csv('/kaggle/input/crime-cast-forecasting-crime-categories/train.csv')
# test_df = pd.read_csv('/kaggle/input/crime-cast-forecasting-crime-categories/test.csv')



# In[41]:


# # Display the first few rows of the training dataset
# print(train_df.head())

# # Check for missing values
# print(train_df.isnull().sum())

# # Data types and summary statistics
# print(train_df.info())
# print(train_df.describe())


# In[42]:


# # Dropping non-numeric columns for correlation matrix
# numeric_df = train_df.select_dtypes(include=[np.number])

# # Correlation matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(numeric_df.corr(), annot=True)
# plt.show()

# # Extracting date and time features
# train_df['Date_Reported'] = pd.to_datetime(train_df['Date_Reported'])
# train_df['Date_Occurred'] = pd.to_datetime(train_df['Date_Occurred'])

# # Example of feature extraction
# train_df['Year_Reported'] = train_df['Date_Reported'].dt.year
# train_df['Month_Reported'] = train_df['Date_Reported'].dt.month
# train_df['Day_Reported'] = train_df['Date_Reported'].dt.day
# train_df['Hour_Occurred'] = train_df['Date_Occurred'].dt.hour
# train_df['Day_Of_Week_Reported'] = train_df['Date_Reported'].dt.dayofweek
# train_df['Day_Of_Week_Occurred'] = train_df['Date_Occurred'].dt.dayofweek

# # Dropping the original Date columns
# train_df = train_df.drop(columns=['Date_Reported', 'Date_Occurred'])

# # Apply the same transformations to the test data
# test_df['Date_Reported'] = pd.to_datetime(test_df['Date_Reported'])
# test_df['Date_Occurred'] = pd.to_datetime(test_df['Date_Occurred'])

# test_df['Year_Reported'] = test_df['Date_Reported'].dt.year
# test_df['Month_Reported'] = test_df['Date_Reported'].dt.month
# test_df['Day_Reported'] = test_df['Date_Reported'].dt.day
# test_df['Hour_Occurred'] = test_df['Date_Occurred'].dt.hour
# test_df['Day_Of_Week_Reported'] = test_df['Date_Reported'].dt.dayofweek
# test_df['Day_Of_Week_Occurred'] = test_df['Date_Occurred'].dt.dayofweek

# test_df = test_df.drop(columns=['Date_Reported', 'Date_Occurred'])

# # Preprocessing pipeline
# numerical_features = ['Latitude', 'Longitude', 'Time_Occurred', 'Victim_Age', 'Year_Reported', 'Month_Reported', 'Day_Reported', 'Hour_Occurred', 'Day_Of_Week_Reported', 'Day_Of_Week_Occurred']
# categorical_features = ['Location', 'Cross_Street', 'Area_Name', 'Victim_Sex', 'Victim_Descent', 'Premise_Description', 'Weapon_Description']

# numerical_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])


# In[43]:


# # Separate features and target variable
# X = train_df.drop(columns=['Crime_Category'])
# y = train_df['Crime_Category']

# # Encode the target variable
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# # Applying the preprocessing steps to the data
# X = preprocessor.fit_transform(X)
# X_test = preprocessor.transform(test_df)

# # Splitting the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# # Define the parameter grid for XGBoost
# xgb_params = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 6, 9],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'subsample': [0.6, 0.8, 1.0]
# }

# # Create the RandomizedSearchCV object
# random_search_xgb = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_distributions=xgb_params, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)

# # Fit the model
# random_search_xgb.fit(X_train, y_train)

# # Best XGBoost Model
# best_xgb = random_search_xgb.best_estimator_
# y_pred_best_xgb = best_xgb.predict(X_val)
# best_xgb_accuracy = accuracy_score(y_val, y_pred_best_xgb)
# print(f'Best XGBoost Accuracy: {best_xgb_accuracy:.4f}')


# In[45]:


# # Predicting on the test set
# test_predictions_best_xgb = best_xgb.predict(X_test)

# # Decode the predictions back to original labels
# test_predictions_best_xgb = label_encoder.inverse_transform(test_predictions_best_xgb)

# # Save predictions to CSV file
# submission_best_xgb = pd.DataFrame({'ID': test_df.index + 1, 'Crime_Category': test_predictions_best_xgb})
# submission_best_xgb.to_csv('submission.csv', index=False)

