import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Dosyamızı yükleyip okuttuk
df = pd.read_csv("C:\projects\intropattern\otu.csv")

df=df.T
# Verileri bir eğitim seti ve bir test seti olarak ayırdık
X = df.iloc[:,1:]
y = df.iloc[:,0]

# Logistic Regression modelimizi oluşturduk
model = LogisticRegression()

# cross_val_score fonksiyonunu kullanarak modelimizi 10 fold cross validation ile test ettik
scores = cross_val_score(model, X, y, cv=10)

# Elde ettiğimiz skorları yazdırdık
print(scores)

# Skorların ortalamasını hesapladık
average_score = sum(scores)/len(scores)
print("Average Score:", average_score)

#Bu örnekte, veri setimiz 10 fold'a böldük ve her bir fold için bir Logistic Regression modeli oluşturduk.
#Model eğitimi ve testi, scikit-learn kütüphanesinin cross_val_score fonksiyonu aracılığıyla otomatik olarak
#yaptık. Böylece, modelimiz veri setimizin tüm elemanları üzerinden eğitilir ve test edilir ve modelin genel 
# performansı daha doğru bir şekilde ölçülebilir.