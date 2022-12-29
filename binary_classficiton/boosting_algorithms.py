import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#datamızı okuduk
df = pd.read_csv("C:\projects\intropattern\otu.csv")
#verinin tranzpozasını alarak left ve rightları column hale getirdik.
df=df.T
X = df.iloc[:,1:]
y = df.iloc[:,0]
#test ve veri setlerimizi ayırdık
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# AdaBoost modelimizi oluşturduk.n_estimators, dosyanın bir parçası olarak eğitilen zayıf modellerin sayısını ifade eder.
# Zayıf bir model, rastgele tahminden yalnızca biraz daha iyi olan ve tipik olarak tek bir güçlü modelden daha az doğru olan bir modeldir.
model = AdaBoostClassifier(n_estimators=100)

# Modeli oluşturduk
model.fit(X_train, y_train)

# Tahminde bulunuyoruz
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)