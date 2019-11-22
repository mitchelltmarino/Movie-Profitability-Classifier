import os
import numpy as np
import pandas as pd
import math

#SKLearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing

# Read in the data.
dataset = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\processed_data_final.csv")

INFORMATIVE_FEATURES = ['Director_ID', 'Director_Name', 'Genres', 'Movie_Title',
                        'Plot_Keywords', 'Content_Rating', 'Movie_ID', 'Release_Date', 
                        'Lead_Actor_ID', 'Lead_Actor_Name', 'Studio_IDs', 'Studio_Names']

# 90% split after lead_actor_ratio
useful = ['Director_Ratio', 'Keywords_Ratio', 'Studios_Ratio', 'Lead_Actor_Ratio',  'Director_Avg_Movie_Revenue', 'Studios_Avg_Movie_Revenue',
'Keywords_Avg_Revenue', 'Lead_Actor_Avg_Movie_Revenue', 'Lead_Actor_Movie_Count', 'Budget']

# Create a new dataframe for target.
target = pd.DataFrame(dataset.pop('Class'), columns=['Class'])

dataset = dataset[useful]
X = dataset.values
y = np.ravel(target.values)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

lr = linear_model.LogisticRegression(max_iter=4000, multi_class='ovr', solver ='liblinear')

# 10 Splits.
NUMBER_OF_SPLITS = 10
kf = KFold(n_splits=NUMBER_OF_SPLITS)
kf.get_n_splits(X)

# prediction_accuracies.
prediction_accuracies = []
confusion_matrices = []

# predictions.
model_predictions = [] 
target_predictions = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model using training sets.
    # Train the model using training sets.
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    prediction_accuracies.append(accuracy)

    # Metrics.
    confusion_matrices.append(metrics.confusion_matrix(y_test, y_pred))
    
    # Predictions and Targets.
    target_predictions.append(y_test)
    model_predictions.append(y_pred)

totals = np.array([[0, 0],[0, 0]])

for matrix in confusion_matrices:

    for i in range(0, 2):
        for j in range(0, 2):
            totals[i][j] += matrix[i][j]

tn, fp, fn, tp = totals.ravel()

cm = np.array([['tn: '+str(tn), 'fp: '+str(fp)],['fn: '+str(fn), 'tp: '+str(tp)]])

print("------------------")
total_records = tn+fp+fn+tp
print("Number of records: ", total_records)
print("------------------")
print("Confusion Matrix:")
print(cm)
print("------------------")
avg = sum(prediction_accuracies)/len(prediction_accuracies)
print("Accuracy: %.9f %%" %(avg*100))
print("------------------")
tpr = (tp/(tp+fn))*100
tnr = (tn/(tn+fp))*100
fpr = (fp/(tn+fp))*100
fnr = (fn/(tp+fn))*100
print("TPR: %.4f %%" %(tpr))
print("TNR: %.4f %%" %(tnr))
print("FPR: %.4f %%" %(fpr))
print("FNR: %.4f %%" %(fnr))
print("------------------")
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
print("Precision: %.4f" %(precision))
print("Recall: %.4f" %(recall))
print("------------------")
f1measure = 2*((precision*recall)/(precision+recall))
print("F1-Measure: %.4f" %(f1measure))


# Predictions
target_predictions = np.concatenate((target_predictions), axis=None)
model_predictions = np.concatenate((model_predictions), axis=None)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(model_predictions, target_predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
lw=2
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Logistic Regression ROC Curve.png')
plt.show()