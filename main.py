import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Credit Score': [720, 650, 500, 680, 695],
    'Annual Income': [80000, 45000, 50000, 60000, 75000],
    'Employment Status': ['Employed', 'Self-Employed', 'Unemployed', 'Employed', 'Employed'],
    'Loan Amount': [5000, 15000, 20000, 10000, 8000],
    'Previous Default': ['No', 'Yes', 'No', 'Yes', 'No'],
    'Loan Approval': ['Approved', 'Denied', 'Denied', 'Denied', 'Approved']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Employment Status', 'Previous Default'])

# Separate features and target variable
X = df_encoded.drop('Loan Approval', axis=1)
y = df_encoded['Loan Approval']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Denied', 'Approved'])
plt.show()

# Print the text representation of the tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)