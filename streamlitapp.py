import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

st.title('Crime Category Prediction')

@st.cache_data
def load_data():
    train_data = pd.read_csv('https://crime-cast-project-60ulvk1f8-gokulanithanandakumars-projects.vercel.app/train.csv')
    test_data = pd.read_csv('https://crime-cast-project-60ulvk1f8-gokulanithanandakumars-projects.vercel.app/test.csv')
    return train_data, test_data

train_data, test_data = load_data()

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

extract_date_features(train_data)
extract_date_features(test_data)

X = train_data.drop(columns=['Crime_Category'])
y = train_data['Crime_Category']

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Classifier': SVC(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Bagging Classifier': BaggingClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=300)
}

results = {}

st.sidebar.title('Select Models to Train')
selected_models = st.sidebar.multiselect('Models', list(models.keys()), default=list(models.keys()))

for model_name in selected_models:
    model = models[model_name]
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    results[model_name] = accuracy

st.write("### Model Performance")
performance_df = pd.DataFrame.from_dict(results, orient='index', columns=['Validation Accuracy'])
st.bar_chart(performance_df)

if 'Logistic Regression' in selected_models:
    param_grid_logistic = {
        'classifier__C': np.logspace(-4, 4, 20),
        'classifier__solver': ['liblinear', 'lbfgs']
    }

    logistic_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=5000))
    ])

    random_search_logistic = RandomizedSearchCV(logistic_pipeline, param_distributions=param_grid_logistic, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_logistic.fit(X_train, y_train)
    logistic_val_accuracy = accuracy_score(y_val, random_search_logistic.predict(X_val))

    st.write(f"Logistic Regression Best Parameters: {random_search_logistic.best_params_}")
    st.write(f"Logistic Regression Validation Accuracy with Best Parameters: {logistic_val_accuracy:.4f}")

# Additional model training and tuning
# Decision Tree
if 'Decision Tree' in selected_models:
    param_grid_decision_tree = {
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    decision_tree_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    random_search_decision_tree = RandomizedSearchCV(decision_tree_pipeline, param_distributions=param_grid_decision_tree, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_decision_tree.fit(X_train, y_train)
    decision_tree_val_accuracy = accuracy_score(y_val, random_search_decision_tree.predict(X_val))

    st.write(f"Decision Tree Best Parameters: {random_search_decision_tree.best_params_}")
    st.write(f"Decision Tree Validation Accuracy with Best Parameters: {decision_tree_val_accuracy:.4f}")

# K-Nearest Neighbors
if 'K-Nearest Neighbors' in selected_models:
    param_grid_knn = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ])
    random_search_knn = RandomizedSearchCV(knn_pipeline, param_distributions=param_grid_knn, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_knn.fit(X_train, y_train)
    knn_val_accuracy = accuracy_score(y_val, random_search_knn.predict(X_val))

    st.write(f"K-Nearest Neighbors Best Parameters: {random_search_knn.best_params_}")
    st.write(f"K-Nearest Neighbors Validation Accuracy with Best Parameters: {knn_val_accuracy:.4f}")

# Support Vector Classifier
if 'Support Vector Classifier' in selected_models:
    param_grid_svc = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': [1, 0.1, 0.01, 0.001],
        'classifier__kernel': ['rbf', 'linear']
    }
    svc_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC())
    ])
    random_search_svc = RandomizedSearchCV(svc_pipeline, param_distributions=param_grid_svc, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_svc.fit(X_train, y_train)
    svc_val_accuracy = accuracy_score(y_val, random_search_svc.predict(X_val))

    st.write(f"Support Vector Classifier Best Parameters: {random_search_svc.best_params_}")
    st.write(f"Support Vector Classifier Validation Accuracy with Best Parameters: {svc_val_accuracy:.4f}")

# Random Forest
if 'Random Forest' in selected_models:
    param_grid_random_forest = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    random_forest_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    random_search_random_forest = RandomizedSearchCV(random_forest_pipeline, param_distributions=param_grid_random_forest, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_random_forest.fit(X_train, y_train)
    random_forest_val_accuracy = accuracy_score(y_val, random_search_random_forest.predict(X_val))

    st.write(f"Random Forest Best Parameters: {random_search_random_forest.best_params_}")
    st.write(f"Random Forest Validation Accuracy with Best Parameters: {random_forest_val_accuracy:.4f}")

# Gradient Boosting
if 'Gradient Boosting' in selected_models:
    param_grid_gradient_boosting = {
        'classifier__n_estimators': [50],
        'classifier__learning_rate': [0.1],
        'classifier__max_depth': [3]
    }
    gradient_boosting_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    random_search_gradient_boosting = RandomizedSearchCV(gradient_boosting_pipeline, param_distributions=param_grid_gradient_boosting, n_iter=3, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_gradient_boosting.fit(X_train, y_train)
    gradient_boosting_val_accuracy = accuracy_score(y_val, random_search_gradient_boosting.predict(X_val))

    st.write(f"Gradient Boosting Best Parameters: {random_search_gradient_boosting.best_params_}")
    st.write(f"Gradient Boosting Validation Accuracy with Best Parameters: {gradient_boosting_val_accuracy:.4f}")

# Bagging Classifier
if 'Bagging Classifier' in selected_models:
    param_grid_bagging = {
        'classifier__n_estimators': [10, 50],
        'classifier__max_samples': [0.5, 1.0],
        'classifier__max_features': [0.5, 1.0]
    }
    bagging_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', BaggingClassifier(random_state=42))
    ])
    random_search_bagging = RandomizedSearchCV(bagging_pipeline, param_distributions=param_grid_bagging, n_iter=3, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_bagging.fit(X_train, y_train)
    bagging_val_accuracy = accuracy_score(y_val, random_search_bagging.predict(X_val))

    st.write(f"Bagging Classifier Best Parameters: {random_search_bagging.best_params_}")
    st.write(f"Bagging Classifier Validation Accuracy with Best Parameters: {bagging_val_accuracy:.4f}")

# Multi-layer Perceptron (MLP)
if 'MLP' in selected_models:
    param_grid_mlp = {
        'classifier__hidden_layer_sizes': [(50,), (100,)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__solver': ['adam'],
        'classifier__alpha': [0.0001]
    }
    mlp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(max_iter=300))
    ])
    random_search_mlp = RandomizedSearchCV(mlp_pipeline, param_distributions=param_grid_mlp, n_iter=3, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search_mlp.fit(X_train, y_train)
    mlp_val_accuracy = accuracy_score(y_val, random_search_mlp.predict(X_val))

    st.write(f"MLP Best Parameters: {random_search_mlp.best_params_}")
    st.write(f"MLP Validation Accuracy with Best Parameters: {mlp_val_accuracy:.4f}")

# Visualizing a sample of the predictions
sample_data = test_data.sample(n=10, random_state=42)
sample_predictions = {}
for model_name in selected_models:
    model = models[model_name]
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    sample_predictions[model_name] = pipeline.predict(sample_data)

sample_df = pd.DataFrame(sample_predictions)
sample_df.index = sample_data.index
sample_df['True_Category'] = train_data.loc[sample_df.index, 'Crime_Category']

st.write("### Sample Predictions")
st.write(sample_df)

# Save final predictions to CSV for the best performing model
best_model_name = performance_df['Validation Accuracy'].idxmax()
best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
final_pipeline.fit(X, y)
final_predictions = final_pipeline.predict(test_data)

submission_df = pd.DataFrame({
    'ID': range(1, len(final_predictions) + 1),
    'Crime_Category': final_predictions
})
submission_df.to_csv('submission.csv', index=False)

st.write(f"Final predictions saved for the best performing model: {best_model_name}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
# from sklearn.neural_network import MLPClassifier
# import warnings
#
# warnings.filterwarnings("ignore")
#
# st.title('Crime Category Prediction')
#
# # File uploaders for training and test data
# uploaded_train_file = st.file_uploader("Upload Training Data CSV", type="csv")
# uploaded_test_file = st.file_uploader("Upload Test Data CSV", type="csv")
#
# if uploaded_train_file is not None:
#     train_data = pd.read_csv(uploaded_train_file)
#     st.write("Training data loaded successfully")
# else:
#     st.write("Please upload the training data file")
#
# if uploaded_test_file is not None:
#     test_data = pd.read_csv(uploaded_test_file)
#     st.write("Test data loaded successfully")
# else:
#     st.write("Please upload the test data file")
#
# # Process data only if both files are uploaded
# if uploaded_train_file is not None and uploaded_test_file is not None:
#     def extract_date_features(df):
#         df['Date_Reported'] = pd.to_datetime(df['Date_Reported'])
#         df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'])
#         df['Year_Reported'] = df['Date_Reported'].dt.year
#         df['Month_Reported'] = df['Date_Reported'].dt.month
#         df['Day_Reported'] = df['Date_Reported'].dt.day
#         df['Hour_Occurred'] = df['Date_Occurred'].dt.hour
#         df['Day_Of_Week_Reported'] = df['Date_Reported'].dt.dayofweek
#         df['Day_Of_Week_Occurred'] = df['Date_Occurred'].dt.dayofweek
#         df.drop(columns=['Date_Reported', 'Date_Occurred'], inplace=True)
#
#
#     extract_date_features(train_data)
#     extract_date_features(test_data)
#
#     X = train_data.drop(columns=['Crime_Category'])
#     y = train_data['Crime_Category']
#
#     num_features = X.select_dtypes(include=['int64', 'float64']).columns
#     cat_features = X.select_dtypes(include=['object']).columns
#
#     num_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#         ('scaler', StandardScaler())
#     ])
#
#     cat_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', num_transformer, num_features),
#             ('cat', cat_transformer, cat_features)
#         ]
#     )
#
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     models = {
#         'Logistic Regression': LogisticRegression(max_iter=5000),
#         'Decision Tree': DecisionTreeClassifier(random_state=42),
#         'K-Nearest Neighbors': KNeighborsClassifier(),
#         'Support Vector Classifier': SVC(),
#         'Random Forest': RandomForestClassifier(random_state=42),
#         'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#         'Bagging Classifier': BaggingClassifier(random_state=42),
#         'MLP': MLPClassifier(max_iter=300)
#     }
#
#     results = {}
#
#     st.sidebar.title('Select Models to Train')
#     selected_models = st.sidebar.multiselect('Models', list(models.keys()), default=list(models.keys()))
#
#     for model_name in selected_models:
#         model = models[model_name]
#         pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', model)
#         ])
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_val)
#         accuracy = accuracy_score(y_val, y_pred)
#         results[model_name] = accuracy
#
#     st.write("### Model Performance")
#     performance_df = pd.DataFrame.from_dict(results, orient='index', columns=['Validation Accuracy'])
#     st.bar_chart(performance_df)
#
#     if 'Logistic Regression' in selected_models:
#         param_grid_logistic = {
#             'classifier__C': np.logspace(-4, 4, 20),
#             'classifier__solver': ['liblinear', 'lbfgs']
#         }
#
#         logistic_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', LogisticRegression(max_iter=5000))
#         ])
#
#         random_search_logistic = RandomizedSearchCV(logistic_pipeline, param_distributions=param_grid_logistic,
#                                                     n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
#         random_search_logistic.fit(X_train, y_train)
#         logistic_val_accuracy = accuracy_score(y_val, random_search_logistic.predict(X_val))
#
#         st.write(f"Logistic Regression Best Parameters: {random_search_logistic.best_params_}")
#         st.write(f"Logistic Regression Validation Accuracy with Best Parameters: {logistic_val_accuracy:.4f}")
#
#     if 'Decision Tree' in selected_models:
#         param_grid_decision_tree = {
#             'classifier__max_depth': [10, 20, 30, None],
#             'classifier__min_samples_split': [2, 5, 10],
#             'classifier__min_samples_leaf': [1, 2, 4]
#         }
#         decision_tree_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', DecisionTreeClassifier(random_state=42))
#         ])
#         random_search_decision_tree = RandomizedSearchCV(decision_tree_pipeline,
#                                                          param_distributions=param_grid_decision_tree, n_iter=10, cv=3,
#                                                          verbose=1, random_state=42, n_jobs=-1)
#         random_search_decision_tree.fit(X_train, y_train)
#         decision_tree_val_accuracy = accuracy_score(y_val, random_search_decision_tree.predict(X_val))
#
#         st.write(f"Decision Tree Best Parameters: {random_search_decision_tree.best_params_}")
#         st.write(f"Decision Tree Validation Accuracy with Best Parameters: {decision_tree_val_accuracy:.4f}")
#
#     if 'K-Nearest Neighbors' in selected_models:
#         param_grid_knn = {
#             'classifier__n_neighbors': [3, 5, 7, 9],
#             'classifier__weights': ['uniform', 'distance'],
#             'classifier__metric': ['euclidean', 'manhattan']
#         }
#         knn_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', KNeighborsClassifier())
#         ])
#         random_search_knn = RandomizedSearchCV(knn_pipeline, param_distributions=param_grid_knn, n_iter=10, cv=3,
#                                                verbose=1, random_state=42, n_jobs=-1)
#         random_search_knn.fit(X_train, y_train)
#         knn_val_accuracy = accuracy_score(y_val, random_search_knn.predict(X_val))
#
#         st.write(f"K-Nearest Neighbors Best Parameters: {random_search_knn.best_params_}")
#         st.write(f"K-Nearest Neighbors Validation Accuracy with Best Parameters: {knn_val_accuracy:.4f}")
#
#     if 'Support Vector Classifier' in selected_models:
#         param_grid_svc = {
#             'classifier__C': [0.1, 1, 10, 100],
#             'classifier__gamma': [1, 0.1, 0.01, 0.001],
#             'classifier__kernel': ['rbf', 'linear']
#         }
#         svc_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', SVC())
#         ])
#         random_search_svc = RandomizedSearchCV(svc_pipeline, param_distributions=param_grid_svc, n_iter=10, cv=3,
#                                                verbose=1, random_state=42, n_jobs=-1)
#         random_search_svc.fit(X_train, y_train)
#         svc_val_accuracy = accuracy_score(y_val, random_search_svc.predict(X_val))
#
#         st.write(f"Support Vector Classifier Best Parameters: {random_search_svc.best_params_}")
#         st.write(f"Support Vector Classifier Validation Accuracy with Best Parameters: {svc_val_accuracy:.4f}")
#
#     if 'Random Forest' in selected_models:
#         param_grid_random_forest = {
#             'classifier__n_estimators': [100, 200, 300],
#             'classifier__max_features': ['auto', 'sqrt', 'log2'],
#             'classifier__max_depth': [10, 20, 30, None],
#             'classifier__min_samples_split': [2, 5, 10],
#             'classifier__min_samples_leaf': [1, 2, 4]
#         }
#         random_forest_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', RandomForestClassifier(random_state=42))
#         ])
#         random_search_random_forest = RandomizedSearchCV(random_forest_pipeline,
#                                                          param_distributions=param_grid_random_forest, n_iter=10, cv=3,
#                                                          verbose=1, random_state=42, n_jobs=-1)
#         random_search_random_forest.fit(X_train, y_train)
#         random_forest_val_accuracy = accuracy_score(y_val, random_search_random_forest.predict(X_val))
#
#         st.write(f"Random Forest Best Parameters: {random_search_random_forest.best_params_}")
#         st.write(f"Random Forest Validation Accuracy with Best Parameters: {random_forest_val_accuracy:.4f}")
#
#     if 'Gradient Boosting' in selected_models:
#         param_grid_gradient_boosting = {
#             'classifier__n_estimators': [100, 200, 300],
#             'classifier__learning_rate': [0.01, 0.1, 0.2],
#             'classifier__max_depth': [3, 5, 7]
#         }
#         gradient_boosting_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', GradientBoostingClassifier(random_state=42))
#         ])
#         random_search_gradient_boosting = RandomizedSearchCV(gradient_boosting_pipeline,
#                                                              param_distributions=param_grid_gradient_boosting,
#                                                              n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
#         random_search_gradient_boosting.fit(X_train, y_train)
#         gradient_boosting_val_accuracy = accuracy_score(y_val, random_search_gradient_boosting.predict(X_val))
#
#         st.write(f"Gradient Boosting Best Parameters: {random_search_gradient_boosting.best_params_}")
#         st.write(f"Gradient Boosting Validation Accuracy with Best Parameters: {gradient_boosting_val_accuracy:.4f}")
#
#     if 'Bagging Classifier' in selected_models:
#         param_grid_bagging = {
#             'classifier__n_estimators': [10, 50, 100],
#             'classifier__base_estimator': [DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression()]
#         }
#         bagging_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', BaggingClassifier(random_state=42))
#         ])
#         random_search_bagging = RandomizedSearchCV(bagging_pipeline, param_distributions=param_grid_bagging, n_iter=10,
#                                                    cv=3, verbose=1, random_state=42, n_jobs=-1)
#         random_search_bagging.fit(X_train, y_train)
#         bagging_val_accuracy = accuracy_score(y_val, random_search_bagging.predict(X_val))
#
#         st.write(f"Bagging Classifier Best Parameters: {random_search_bagging.best_params_}")
#         st.write(f"Bagging Classifier Validation Accuracy with Best Parameters: {bagging_val_accuracy:.4f}")
#
#     if 'MLP' in selected_models:
#         param_grid_mlp = {
#             'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
#             'classifier__activation': ['relu', 'tanh'],
#             'classifier__solver': ['adam', 'sgd'],
#             'classifier__alpha': [0.0001, 0.001, 0.01]
#         }
#         mlp_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', MLPClassifier(max_iter=300))
#         ])
#         random_search_mlp = RandomizedSearchCV(mlp_pipeline, param_distributions=param_grid_mlp, n_iter=10, cv=3,
#                                                verbose=1, random_state=42, n_jobs=-1)
#         random_search_mlp.fit(X_train, y_train)
#         mlp_val_accuracy = accuracy_score(y_val, random_search_mlp.predict(X_val))
#
#         st.write(f"MLP Best Parameters: {random_search_mlp.best_params_}")
#         st.write(f"MLP Validation Accuracy with Best Parameters: {mlp_val_accuracy:.4f}")
