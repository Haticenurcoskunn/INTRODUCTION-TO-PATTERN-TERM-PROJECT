import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Dosyamızı yükleyip okuttuk
df = pd.read_csv("C:\projects\intropattern\otu.csv")
#verinin tranzpozasını alarak left ve rightları column hale getirdik.
df=df.T
X = df.iloc[:,1:]
y = df.iloc[:,0]
#Verileri bir eğitim seti ve bir test seti olarak ayırdık
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=0), df["target"], test_size=0.2)

# Bir karar ağacı modeli oluşturduk
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
#Modelin doğruluğunu hesapladık
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)