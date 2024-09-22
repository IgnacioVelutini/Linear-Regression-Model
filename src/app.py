from utils import db_connect
engine = db_connect()

# your code here


# app.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv"
data = pd.read_csv(url)

# Step 2: Perform Exploratory Data Analysis (EDA)
def perform_eda(data):
    # Drop columns with more than 80% missing values
    data = data.dropna(axis=1, thresh=0.8 * len(data))

    # Drop rows with missing values
    data = data.dropna()

    # Keep only numeric columns
    data = data.select_dtypes(include=[float, int])

    return data

data = perform_eda(data)

# Step 3: Build Linear Regression Model
def build_linear_regression(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Define features (X) and target (y)
X = data.drop('Heart disease_number', axis=1)
y = data['Heart disease_number']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate Linear Regression model
linear_model = build_linear_regression(X_train, y_train)
mse, r2 = evaluate_model(linear_model, X_test, y_test)

print(f"Linear Regression Model - Mean Squared Error: {mse}, R^2 Score: {r2}")

# Step 4: Build and Evaluate Lasso Regression Model
def build_lasso_model(X_train, y_train, X_test, y_test, alphas):
    r2_scores = []
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        y_pred = lasso_model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
    return r2_scores

# Test Lasso model with different alpha values
alphas = [0.1, 1, 5, 10, 20]
r2_scores_lasso = build_lasso_model(X_train, y_train, X_test, y_test, alphas)

# Plot the R^2 scores
plt.plot(alphas, r2_scores_lasso)
plt.xlabel('Alpha')
plt.ylabel('R^2 Score')
plt.title('R^2 Score vs Alpha in Lasso Regression')
plt.show()

# Step 5: Optimize Lasso Model using GridSearchCV
def optimize_lasso(X_train, y_train, X_test, y_test):
    param_grid = {'alpha': [0.1, 1, 5, 10, 20]}
    lasso_cv = GridSearchCV(Lasso(), param_grid, cv=5)
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_
    best_r2 = r2_score(y_test, best_lasso.predict(X_test))
    return best_lasso, best_r2

# Optimize Lasso model
best_lasso_model, best_r2 = optimize_lasso(X_train, y_train, X_test, y_test)
print(f"Optimized Lasso Model - Best R^2 Score: {best_r2}")
