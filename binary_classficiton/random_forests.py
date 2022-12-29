import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df= pd.read_csv("C:\projects\intropattern\otu.csv")
#verinin tranzpozasını alarak left ve rightları column hale getirdik.
df=df.T

X= df.iloc[:,1:]
y = df.iloc[:,0]
#Verileri bir eğitim seti ve bir test seti olarak ayırdık
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
# Model oluştur (örnek olarak 100 adet karar ağacı kullanılacağız)
model = RandomForestClassifier(n_estimators=100)
# Model uyguluyoruz
model.fit(X_train, y_train)
# Tahminleri yaptık
predictions = model.predict(X_test)

# Performansı değerlendirdik
print(accuracy_score(y_test, predictions))  

# Model eğittik
model.fit(X_train, y_train)