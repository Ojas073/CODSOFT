import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


train_data = pd.read_csv("train_data.txt", sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv("test_data.txt", sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_solution_data = pd.read_csv("test_data_solution.txt", sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])


plt.figure(figsize=(20, 8))
sns.countplot(y=train_data['GENRE'], order=train_data['GENRE'].value_counts().index)
plt.title('Number of Movies per Genre')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()


X = train_data['DESCRIPTION']
y = train_data['GENRE']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)


model = LinearSVC()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))


test_X = vectorizer.transform(test_data['DESCRIPTION'].astype(str))
test_preds = model.predict(test_X)
test_preds_labels = le.inverse_transform(test_preds)


output_df = test_data.copy()
output_df['PREDICTED_GENRE'] = test_preds_labels
output_df.to_csv("predicted_test_data.csv", index=False)
print("Predictions saved to predicted_test_data.csv")
