
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv("creditcard.csv")

# Print initial data information
print(data.shape)
print(data.describe())

# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud) / float(len(valid))
print(f'Outlier fraction: {outlierFraction}')
print(f'Fraud Cases: {len(fraud)}')
print(f'Valid Transactions: {len(valid)}')

# Explore the details of fraudulent and valid transactions
print("Amount details of the fraudulent transaction")
print(fraud.Amount.describe())

print("Details of valid transaction")
print(valid.Amount.describe())

# Correlation matrix
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Preparing the data for modeling
X = data.drop(['Class'], axis=1)
Y = data['Class']
xData = X.values
yData = Y.values

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
sm = SMOTE(random_state=42)
xTrain_res, yTrain_res = sm.fit_resample(xTrain, yTrain)

# Train the Random Forest model with resampled data
rfc = RandomForestClassifier()
rfc.fit(xTrain_res, yTrain_res)

# Predictions
yPred = rfc.predict(xTest)

# Evaluate the classifier
print("The model used is Random Forest classifier")
print(f"The accuracy is {accuracy_score(yTest, yPred)}")
print(f"The precision is {precision_score(yTest, yPred)}")
print(f"The recall is {recall_score(yTest, yPred)}")
print(f"The F1-Score is {f1_score(yTest, yPred)}")
print(f"The Matthews correlation coefficient is {matthews_corrcoef(yTest, yPred)}")

# Print the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
