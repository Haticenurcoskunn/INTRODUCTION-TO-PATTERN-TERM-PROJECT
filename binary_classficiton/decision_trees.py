import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#* 
#!pip install matplotlib
import matplotlib.pyplot as plt
def accuracyYazdir(acc,pngName):

  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(acc, acc, 'bo', label='Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Accuracy')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(pngName+'.png', dpi=300, bbox_inches='tight')
  plt.show()
  
def confMatrix(cf_matrix,pngName):
  from sklearn.metrics import confusion_matrix
  import sklearn.metrics as mt
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
  ax.set_title('Seaborn Confusion Matrix with labels\n\n');
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values ');
  ## Ticket labels - List must be in alphabetical order
  ax.xaxis.set_ticklabels(['False','True'])
  ax.yaxis.set_ticklabels(['False','True'])
  plt.legend()
  ## Display the visualization of the Confusion Matrix.
  plt.savefig(pngName+'.png', dpi=300, bbox_inches='tight')
  plt.show()

#*

# Dosyamızı yükleyip okuttuk
df = pd.read_csv("../otu.csv")
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
#*
#pred_bnr=   (model.predict(x_test) >= 0.5).astype("int")
#conf = confusion_matrix(y_test, pred_bnr)
#confMatrix(conf,"../decision_trees")
#accuracyYazdir(accuracy,"../acc_decision_trees")
#*
print("Accuracy:", accuracy)
