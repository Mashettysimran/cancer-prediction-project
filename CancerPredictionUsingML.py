# Test case:- From the given data Wbc.csv, we need to predict the presence of cancer
# Information about the data:-
# This CSV file contains a test report on different white blood cells, based on which to predict whether cancer is present or not**
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
df=pd.read_csv(r"/content/drive/MyDrive/wbc.csv")
print(df)
print(df.columns)
df.drop(columns=["id",'Unnamed: 32'],inplace=True)
# **data jar 1)clean 2)encode 3)split 4) scale**
sns.scatterplot(data=df,x="radius_mean",y="texture_mean",hue="diagnosis")
plt.show()
df.info()
df.shape
df.isnull().sum()
df.duplicated().sum()
df["diagnosis"].unique()
df["diagnosis"].value_counts()
df.describe()
#outliers detection
num_columns=df.select_dtypes(include=["int64","float64"]).columns
for col in num_columns:
  q1=df[col].quantile(0.25)
  q3=df[col].quantile(0.75)
  iqr=q3-q1
  l_b=q1-1.5*iqr
  u_b=q3+1.5*iqr
  outliers = df[(df[col] < l_b) | (df[col] > u_b)]
  print(col, len(outliers))
df.shape
# FEATURES AND LABEL
x = df.iloc[:,1:]
y = df["diagnosis"]
# ENCODING TARGET
y = y.map({"M":1,"B":0})
# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=45)
# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model=LogisticRegression()
model.fit(x_train,y_train)
#predicting using testing data
y_pred=model.predict(x_test)
y_pred
d=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(d)
#evaluating the model
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,precision_score,recall_score
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.show()
from sklearn.neighbors import KNeighborsClassifier
sns.scatterplot(data=df,x="radius_mean",y="texture_mean",hue="diagnosis")
plt.show()
from sklearn.model_selection import cross_val_score
# FEATURES AND LABEL
x = df.iloc[:,1:]
y = df["diagnosis"]
# ENCODING THE TARGET
y = y.map({"M":1,"B":0})
# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=45)
# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
for i in(1,2,3,4,5,6,7,20,30):
  knn=KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  #train_score=knn.score(x_train,y_train)
  #cv_score=np.mean(cross_val_score(knn,x_train,y_train,cv=10))
  print("kvalue",i,"train_score:",knn.score(x_train,y_train),"cv_score:",np.mean(cross_val_score(knn,x_train,y_train,cv=10)))
## The best K value in Machine Learning is selected based on the highest cross-validation score with minimal difference between training and validation accuracy, indicating good generalisation and reduced overfitting.  
import matplotlib.pyplot as plt
k_values = [1,2,3,4,5,6,7,20,30]
train_scores = [1.0,0.9670,0.9802,0.9758,0.9780,0.9736,0.9780,0.9582,0.9538]
cv_scores = [0.9515,0.9602,0.9735,0.9669,0.9713,0.9646,0.9690,0.9560,0.9494]
plt.plot(k_values, train_scores, marker='o', label="Train Score")
plt.plot(k_values, cv_scores, marker='o', label="CV Score")
plt.xlabel("K Value")
plt.ylabel("Score")
plt.title("KNN Performance")
plt.legend()
plt.show()  
# so we can conclude that the best k value is k=3
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pre=knn.predict(x_test)
y_pre
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,precision_score,recall_score
print(accuracy_score(y_test,y_pre))
print(f1_score(y_test,y_pre))
print(precision_score(y_test,y_pre))
print(recall_score(y_test,y_pre))
# | Metric    | Logistic Regression | KNN    |
# | --------- | ------------------- | ------ |
# | Accuracy  | **98.24%**          | 95.61% |
# | F1-Score  | **97.56%**          | 93.82% |
# | Precision | **100%**            | 97.43% |
# | Recall    | **95.23%**          | 90.47% |
# Both models achieved strong performance in breast cancer prediction. However, Logistic Regression outperformed KNN in all major evaluation metrics.
# The model demonstrated excellent precision, recall, and overall classification accuracy, making it the most suitable model for this dataset.
# Hence, Logistic Regression was selected as the final and best-performing model for breast cancer classification.
