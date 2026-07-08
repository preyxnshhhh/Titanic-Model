# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

if 'Survived' not in df_test.columns:
    df_test['Survived'] = 0

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess(df_train, df_test):
    df = pd.concat([df_train, df_test], axis=0)

    # Drop unnecessary
    df = df.drop(['Name', 'Ticket'], axis=1)

    # Fill missing
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Cabin'] = df['Cabin'].fillna('X000')
    df['Embarked'] = df['Embarked'].fillna('X')
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    # Extract cabin info
    df['cabin_letter'] = df['Cabin'].str.extract(r'([A-Za-z]+)', expand=False)
    df['cabin_number'] = df['Cabin'].str.extract(r'(\d+)', expand=False)

    df = df.drop('Cabin', axis=1)

    # Encoding
    df = pd.get_dummies(df, columns=['cabin_letter'], prefix='cabin')
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
    df = pd.get_dummies(df, columns=['Sex'], prefix='Sex')

    # Drop unwanted dummies
    if 'cabin_X' in df.columns:
        df = df.drop('cabin_X', axis=1)
    if 'Embarked_X' in df.columns:
        df = df.drop('Embarked_X', axis=1)

    # Convert cabin number
    df['cabin_number'] = df['cabin_number'].fillna(0)
    df['cabin_number'] = pd.to_numeric(df['cabin_number'])

    # Feature Engineering
    df['Pclass_bin_Fare'] = df['Fare'] // df['Pclass']
    df['Pclass_bin_sex'] = df['Pclass'] - df['Sex_female']

    # Split back
    df_train = df[:len(df_train)]
    df_test = df[len(df_train):]

    df_test = df_test.drop('Survived', axis=1)

    return df_train, df_test

# Apply preprocessing
train_df, test_df = preprocess(df_train, df_test)

# =========================
# SPLIT DATA
# =========================
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.values.ravel()

# =========================
# MODELS
# =========================

# Logistic Regression (with imbalance handling)
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
model_lr.fit(X_train, y_train)

# Random Forest (best performing)
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("\n=== ACCURACY ===")
print("Logistic:", accuracy_score(y_test, model_lr.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, model_rf.predict(X_test)))

# =========================
# THRESHOLD TUNING
# =========================
print("\n=== THRESHOLD TUNING (Logistic) ===")

probs = model_lr.predict_proba(X_test)[:,1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for t in thresholds:
    preds = (probs >= t).astype(int)
    print(f"\nThreshold: {t}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))

# =========================
# ROC CURVE
# =========================
probs_lr = model_lr.predict_proba(X_test)[:,1]
probs_rf = model_rf.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, probs_rf)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"Logistic AUC = {auc_lr:.2f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC = {auc_rf:.2f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# FINAL PREDICTION (USE RF)
# =========================
probs_test = model_rf.predict_proba(test_df)[:,1]

# You can tune this threshold (0.5 default)
final_pred = (probs_test >= 0.5).astype(int)

# Save output
final = pd.DataFrame()
final['PassengerId'] = test_df['PassengerId']
final['Survived'] = final_pred

final.to_csv("output.csv", index=False)

print("\n✅ Output saved as output.csv")
