# Machine-Learning
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score
)

# Set random seed for reproducibility
RANDOM_STATE = 42

def evaluate_model(y_test, y_pred, y_proba=None):
    """Utility function to display model evaluation metrics."""
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report (Precision, Recall, F1-Score):\n", classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# ==========================================
# Task 1: Random Forest on Breast Cancer Dataset
# ==========================================
print("=================== TASK 1: Random Forest Classifier ===================")

# Load Breast Cancer dataset from sklearn
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target

# Train-Test Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=RANDOM_STATE
)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=RANDOM_STATE)
rf_clf.fit(X_train_c, y_train_c)

# Predict & Evaluate
y_pred_rf = rf_clf.predict(X_test_c)
y_proba_rf = rf_clf.predict_proba(X_test_c)[:, 1]

evaluate_model(y_test_c, y_pred_rf, y_proba_rf)


# ==========================================
# Task 2: Logistic Regression on Diabetes Dataset
# ==========================================
print("\n=================== TASK 2: Logistic Regression ===================")

# 1. Load the dataset & 2. Assign column names
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df_diabetes = pd.read_csv(diabetes_url, names=diabetes_cols)

# 3. Check for missing or zero values (handling invalid 0 values)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_diabetes[zero_cols] = df_diabetes[zero_cols].replace(0, np.nan)
df_diabetes.fillna(df_diabetes.median(), inplace=True)

# 4. Split dataset into train (80%) and test (20%)
X_diab = df_diabetes.drop('Outcome', axis=1)
y_diab = df_diabetes['Outcome']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=RANDOM_STATE
)

# 5. Apply feature scaling
scaler_d = StandardScaler()
X_train_d_scaled = scaler_d.fit_transform(X_train_d)
X_test_d_scaled = scaler_d.transform(X_test_d)

# 6. Train Logistic Regression model
log_reg = LogisticRegression(random_state=RANDOM_STATE)
log_reg.fit(X_train_d_scaled, y_train_d)

# 7. Evaluate Model
y_pred_lr = log_reg.predict(X_test_d_scaled)
y_proba_lr = log_reg.predict_proba(X_test_d_scaled)[:, 1]

evaluate_model(y_test_d, y_pred_lr, y_proba_lr)

# 8. Interpret Model Coefficients
print("\nModel Coefficients:")
coef_df = pd.DataFrame({'Feature': X_diab.columns, 'Coefficient': log_reg.coef_[0]})
print(coef_df.sort_values(by='Coefficient', ascending=False))


# ==========================================
# Task 3: XGBoost Classifier on Titanic Dataset
# ==========================================
print("\n=================== TASK 3: XGBoost Classifier ===================")

# 1. Load dataset & 2. Assign column names
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_titanic = pd.read_csv(titanic_url)

# Select relevant columns and clean missing values
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df_titanic['Sex'] = df_titanic['Sex'].map({'male': 0, 'female': 1})
df_titanic['Age'].fillna(df_titanic['Age'].median(), inplace=True)
df_titanic['Fare'].fillna(df_titanic['Fare'].median(), inplace=True)

X_tita = df_titanic[features]
y_tita = df_titanic['Survived']

# 4. Split dataset into train (80%) and test (20%)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tita, y_tita, test_size=0.2, random_state=RANDOM_STATE
)

# 5. Apply feature scaling
scaler_t = StandardScaler()
X_train_t_scaled = scaler_t.fit_transform(X_train_t)
X_test_t_scaled = scaler_t.transform(X_test_t)

# 6. Train XGBoost model with parameters
xgb_clf = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=3, 
    random_state=RANDOM_STATE, 
    eval_metric='logloss'
)
xgb_clf.fit(X_train_t_scaled, y_train_t)

# 7. Evaluate Model
y_pred_xgb = xgb_clf.predict(X_test_t_scaled)
y_proba_xgb = xgb_clf.predict_proba(X_test_t_scaled)[:, 1]

evaluate_model(y_test_t, y_pred_xgb, y_proba_xgb)

# 8. Interpret Feature Importances (Model Coefficients analog for Trees)
print("\nFeature Importances:")
feat_imp_xgb = pd.DataFrame({'Feature': features, 'Importance': xgb_clf.feature_importances_})
print(feat_imp_xgb.sort_values(by='Importance', ascending=False))


# ==========================================
# Task 4: Decision Tree Classifier on Pima Diabetes
# ==========================================
print("\n=================== TASK 4: Decision Tree Classifier ===================")

# 1-3. Load and clean Pima Diabetes dataset
df_pima = pd.read_csv(diabetes_url, names=diabetes_cols)
df_pima[zero_cols] = df_pima[zero_cols].replace(0, np.nan)
df_pima.fillna(df_pima.median(), inplace=True)

# 4-5. Define X, y and split 80-20 with random_state=42
X_pima = df_pima.drop('Outcome', axis=1)
y_pima = df_pima['Outcome']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=42
)

# 6. Train Default DecisionTreeClassifier
dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train_p, y_train_p)

print("\n--- Unconstrained Decision Tree ---")
y_pred_dt1 = dt_unconstrained.predict(X_test_p)
evaluate_model(y_test_p, y_pred_dt1)

# 8. Train Restricted DecisionTree (max_depth=3)
dt_restricted = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_restricted.fit(X_train_p, y_train_p)


print("\n--- Restricted Decision Tree (max_depth=3) ---")
y_pred_dt2 = dt_restricted.predict(X_test_p)
evaluate_model(y_test_p, y_pred_dt2)

# Extract and display feature importances for restricted tree
print("\nFeature Importances (Restricted Tree):")
dt_imp = pd.DataFrame({'Feature': X_pima.columns, 'Importance': dt_restricted.feature_importances_})
print(dt_imp.sort_values(by='Importance', ascending=False))
pip install numpy pandas scikit-learn xgboost
git init
git add .
git commit -m "Add ML models implementation"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
