import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics

data = pd.read_excel(r'iris.xls')
df= pd.DataFrame(data, columns= ['SL','SW','PL','PW','Classification'])
y = df['Classification'] 
y = y.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [0,1,2])
x = df.drop(['Classification'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size=0.33)

logireg = LogisticRegression()
logireg.fit(x,y)
logireg_y_pred=logireg.predict(X_test)
logireg_accuracy=metrics.accuracy_score(y_test,logireg_y_pred)

DTC = DecisionTreeClassifier(random_state=42)
DTC.fit(x,y)
DTC_y_pred=DTC.predict(X_test)
DTC_accuracy=metrics.accuracy_score(y_test,DTC_y_pred)

GNB = GaussianNB()
GNB.fit(x,y)
GNB_y_pred=GNB.predict(X_test)
GNB_accuracy=metrics.accuracy_score(y_test,GNB_y_pred)

RFC=RandomForestClassifier(n_estimators=100)
RFC.fit(x,y)
RFC_y_pred=RFC.predict(X_test)
RFC_accuracy=metrics.accuracy_score(y_test,RFC_y_pred)

KNN = KNeighborsClassifier(n_neighbors=15)
KNN.fit(x,y)
KNN_y_pred=KNN.predict(X_test)
KNN_accuracy=metrics.accuracy_score(y_test,KNN_y_pred)

SVM = SVC(random_state=42)
SVM.fit(x,y)
SVM_y_pred=SVM.predict(X_test)
SVM_accuracy=metrics.accuracy_score(y_test,SVM_y_pred)

SGD = SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
SGD.fit(x,y)
SGD_y_pred=SGD.predict(X_test)
SGD_accuracy=metrics.accuracy_score(y_test,SGD_y_pred)

pred={
      'logireg':logireg,
      'logireg_accuracy':logireg_accuracy,
      'DTC':DTC,
      'DTC_accuracy':DTC_accuracy,
      'GNB':GNB,
      'GNB_accuracy':GNB_accuracy,
      'RFC':RFC,
      'RFC_accuracy':RFC_accuracy,
      'KNN':KNN,
      'KNN_accuracy':KNN_accuracy,
      'SVM':SVM,
      'SVM_accuracy':SVM_accuracy,
      'SGD':SGD,
      'SGD_accuracy':SGD_accuracy,
      }

pickle.dump(pred,open('model.pkl','wb'))