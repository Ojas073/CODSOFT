import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


train = pd.read_csv("data/fraudTrain.csv")
test = pd.read_csv("data/fraudTest.csv")


data = pd.concat([train, test])


drop_cols = [
    'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state',
    'job', 'dob', 'unix_time', 'merchant', 'merch_lat', 'merch_long'
]
data.drop(columns=drop_cols, inplace=True)


data = pd.get_dummies(data, columns=['category', 'gender'], drop_first=True)


X = data.drop('is_fraud', axis=1)
X = X.select_dtypes(include=[np.number]) 
y = data['is_fraud']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(" Classification Report:\n")
print(classification_report(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()


os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/confusion_matrix.png")
print(" Confusion matrix saved to visuals/confusion_matrix.png")
