import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sn

df = pd.read_csv("C:\projects\intropattern\otu.csv")
#verinin tranzpozasını alarak left ve rightları column hale getirdik.
df=df.T
X = df.iloc[:,1:]
y = df.iloc[:,0]
#Verileri bir eğitim seti ve bir test seti olarak ayırdık
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
print (X_test)
print (y_pred)
confusion_matrix = pd.crosstab(y_pred, y_test, rownames=['Sample 1'], colnames=['right'])
sn.heatmap(confusion_matrix, annot=True)
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
print('Doğruluk : ', accuracy)
print("Test Doğruluk Orani (%) : ", accuracy_percentage)






























