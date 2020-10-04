#Decision tree algorithim
#importing linraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

#importing dataset
dataset = pd.read_csv('datasets_19_420_Iris.csv')
dataset.isnull().sum()#Checking for missing values
x = dataset.iloc[:,1:5]
y = dataset.iloc[:,-1]

#Spliting dataset in to trainning and testing set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state = 0)

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_x =StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#Fitting Decision Tree Classification  to Trainning set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state =0)
classifier.fit(x_train, y_train)

#Pridicting the test result
y_pred = classifier.predict(x_test)
y_pred = pd.DataFrame(y_pred)
#Model accuracy
from sklearn import metrics
print("Model accuracy is :",metrics.accuracy_score(y_test, y_pred))
# cheching test accuracy using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred )
names = np.unique(y_pred)
sns.heatmap(cm, square=True, annot= True, fmt='d',cbar =False, xticklabels=names, yticklabels=names)
plt.xlabel('True Value')
plt.ylabel('Predicted Values by Machine')
plt.savefig('Decision tree algorithim accuracy.png')
