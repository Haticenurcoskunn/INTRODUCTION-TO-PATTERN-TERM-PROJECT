import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# Dosyamızı yükleyip okuttuk
df = pd.read_csv("C:\projects\intropattern\otu.csv")

df=df.T
# Verileri bir eğitim seti ve bir test seti olarak ayırdık
X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic Regression modelimizi oluşturduk
model = LogisticRegression()
model.fit(X_train, y_train)

# Modelimizi test ettik
y_pred = model.predict(X_test)

# AUC metrikini hesapladık
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)

# Confusion matrix metriklerini hesapladık
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix)