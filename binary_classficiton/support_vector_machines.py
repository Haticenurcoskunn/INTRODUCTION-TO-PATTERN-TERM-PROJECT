import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Dosyamızı yükleyip okuttuk
data=pd.read_csv('C:\projects\intropattern\otu.csv',low_memory=False)

#verinin tranzpozasını alarak left ve rightları column hale getirdik.
data=data.T

X= data.iloc[:,1:]
y= data.iloc[:,0]
#Verileri bir eğitim seti ve bir test seti olarak ayırdık
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SVM modeli oluşturduk
svm_model= SVC(kernel="linear")

svm_model.fit(X_train,y_train)
#Ayırdığımız test setimizi (X_test) kullanarak oluşturduğumuz model ile tahmin yaptık
#ve elde ettiğimiz set (y_pred) ile hedef değişken (y_test) test setimizi karşılaştıralım.
y_pred=svm_model.predict(X_test)
accuracy =accuracy_score(X_test, y_test)
print('Test doğruluk orani:', accuracy)