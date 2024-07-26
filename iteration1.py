import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

# Load the datasets
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Extract date and time features
def extract_date_features(df):
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'])
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'])
    df['Year_Reported'] = df['Date_Reported'].dt.year
    df['Month_Reported'] = df['Date_Reported'].dt.month
    df['Day_Reported'] = df['Date_Reported'].dt.day
    df['Hour_Occurred'] = df['Date_Occurred'].dt.hour
    df['Minute_Occurred'] = df['Date_Occurred'].dt.minute
    df['Second_Occurred'] = df['Date_Occurred'].dt.second
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

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model: Logistic Regression with basic settings
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))
])

baseline_pipeline.fit(X_train, y_train)
baseline_val_accuracy = accuracy_score(y_val, baseline_pipeline.predict(X_val))
print(f"Baseline Logistic Regression Validation Accuracy: {baseline_val_accuracy:.4f}")

# Logistic Regression with Grid Search
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))
])

param_grid_logistic = {
    'classifier__C': np.logspace(-4, 4, 20),
    'classifier__solver': ['liblinear', 'lbfgs']
}

grid_search_logistic = GridSearchCV(logistic_pipeline, param_grid=param_grid_logistic, cv=cv, verbose=1, n_jobs=-1)
grid_search_logistic.fit(X_train, y_train)
logistic_val_accuracy = accuracy_score(y_val, grid_search_logistic.predict(X_val))
print(f"Logistic Regression Validation Accuracy: {logistic_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_logistic.best_params_}")

# Decision Tree with Grid Search
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

param_grid_decision_tree = {
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search_decision_tree = GridSearchCV(decision_tree_pipeline, param_grid=param_grid_decision_tree, cv=cv, verbose=1, n_jobs=-1)
grid_search_decision_tree.fit(X_train, y_train)
decision_tree_val_accuracy = accuracy_score(y_val, grid_search_decision_tree.predict(X_val))
print(f"Decision Tree Validation Accuracy: {decision_tree_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_decision_tree.best_params_}")

# K-Nearest Neighbors with Grid Search
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

grid_search_knn = GridSearchCV(knn_pipeline, param_grid=param_grid_knn, cv=cv, verbose=1, n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
knn_val_accuracy = accuracy_score(y_val, grid_search_knn.predict(X_val))
print(f"K-Nearest Neighbors Validation Accuracy: {knn_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_knn.best_params_}")

# Support Vector Classifier with Grid Search
svc_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

param_grid_svc = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear']
}

grid_search_svc = GridSearchCV(svc_pipeline, param_grid=param_grid_svc, cv=cv, verbose=1, n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
svc_val_accuracy = accuracy_score(y_val, grid_search_svc.predict(X_val))
print(f"Support Vector Classifier Validation Accuracy: {svc_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_svc.best_params_}")

# Random Forest with Grid Search
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

grid_search_random_forest = GridSearchCV(random_forest_pipeline, param_grid=param_grid_random_forest, cv=cv, verbose=1, n_jobs=-1)
grid_search_random_forest.fit(X_train, y_train)
random_forest_val_accuracy = accuracy_score(y_val, grid_search_random_forest.predict(X_val))
print(f"Random Forest Validation Accuracy: {random_forest_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_random_forest.best_params_}")

# Bagging Classifier with Grid Search
bagging_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BaggingClassifier(random_state=42))
])

param_grid_bagging = {
    'classifier__n_estimators': [10, 50],
    'classifier__max_samples': [0.5, 1.0],
    'classifier__max_features': [0.5, 1.0]
}

grid_search_bagging = GridSearchCV(bagging_pipeline, param_grid=param_grid_bagging, cv=cv, verbose=1, n_jobs=-1)
grid_search_bagging.fit(X_train, y_train)
bagging_val_accuracy = accuracy_score(y_val, grid_search_bagging.predict(X_val))
print(f"Bagging Classifier Validation Accuracy: {bagging_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_bagging.best_params_}")

# Gradient Boosting with Grid Search
gradient_boosting_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

param_grid_gradient_boosting = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.1, 0.01],
    'classifier__max_depth': [3, 5, 7]
}

grid_search_gradient_boosting = GridSearchCV(gradient_boosting_pipeline, param_grid=param_grid_gradient_boosting, cv=cv, verbose=1, n_jobs=-1)
grid_search_gradient_boosting.fit(X_train, y_train)
gradient_boosting_val_accuracy = accuracy_score(y_val, grid_search_gradient_boosting.predict(X_val))
print(f"Gradient Boosting Validation Accuracy: {gradient_boosting_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_gradient_boosting.best_params_}")

# Neural Network with Grid Search
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(max_iter=1000, random_state=42))
])

param_grid_mlp = {
    'classifier__hidden_layer_sizes': [(50, 50), (100,)],
    'classifier__activation': ['tanh', 'relu'],
    'classifier__solver': ['adam', 'sgd']
}

grid_search_mlp = GridSearchCV(mlp_pipeline, param_grid=param_grid_mlp, cv=cv, verbose=1, n_jobs=-1)
grid_search_mlp.fit(X_train, y_train)
mlp_val_accuracy = accuracy_score(y_val, grid_search_mlp.predict(X_val))
print(f"Neural Network Validation Accuracy: {mlp_val_accuracy:.4f}")
print(f"Best Parameters: {grid_search_mlp.best_params_}")

# Stacking Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(C=1, gamma=0.1, kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'))
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=5000))
stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

stacking_pipeline.fit(X_train, y_train)
stacking_val_accuracy = accuracy_score(y_val, stacking_pipeline.predict(X_val))
print(f"Stacking Classifier Validation Accuracy: {stacking_val_accuracy:.4f}")

# Choose the best model based on validation accuracy
best_model = stacking_pipeline if stacking_val_accuracy > max(logistic_val_accuracy, decision_tree_val_accuracy, knn_val_accuracy, svc_val_accuracy, random_forest_val_accuracy, bagging_val_accuracy, gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_logistic if logistic_val_accuracy >= max(decision_tree_val_accuracy, knn_val_accuracy, svc_val_accuracy, random_forest_val_accuracy, bagging_val_accuracy, gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_decision_tree if decision_tree_val_accuracy >= max(knn_val_accuracy, svc_val_accuracy, random_forest_val_accuracy, bagging_val_accuracy, gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_knn if knn_val_accuracy >= max(svc_val_accuracy, random_forest_val_accuracy, bagging_val_accuracy, gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_svc if svc_val_accuracy >= max(random_forest_val_accuracy, bagging_val_accuracy, gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_random_forest if random_forest_val_accuracy >= max(bagging_val_accuracy, gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_bagging if bagging_val_accuracy >= max(gradient_boosting_val_accuracy, mlp_val_accuracy) else grid_search_gradient_boosting if gradient_boosting_val_accuracy >= mlp_val_accuracy else grid_search_mlp

# Predict on test data with the best model
test_predictions = best_model.predict(test_data)

# Create a DataFrame to save the predictions
submission = pd.DataFrame({
    'Id': test_data.index,
    'Crime_Category': test_predictions
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file created.")
