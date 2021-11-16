from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as met 
from sklearn import preprocessing

cancer = pd.read_csv('breast-cancer.data',sep=",") 
df = pd.DataFrame(cancer)
print(df.head())

le = preprocessing.LabelEncoder()
id3_model = DecisionTreeClassifier(criterion = 'entropy') 
cart_model = DecisionTreeClassifier(criterion = 'gini') 
print(id3_model)
print(cart_model)

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
for i in range(9): 
    le.fit(list(set(df.iloc[:,i].values))) 
    a =[]
    for j in range(len(X)): 
        a.append(X[j][i])
    a = le.transform(a)
    for j in range(len(X)): 
        X[j][i] = a[j]
le.fit(list(set(df.iloc[:,-1].values))) 
a =[]

for j in range(len(X)): 
    a.append(Y[j])
a = le.transform(a)
for j in range(len(X)): 
    Y[j] = a[j]
print(X)  
print(Y) 
Y=Y.astype('int')
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size =int(0.1*len(Y)), random_state = 1)
print(Xtrain) 
print(Ytrain)

id3_model_trained = id3_model.fit(Xtrain,Ytrain) 
cart_model_trained = cart_model.fit(Xtrain,Ytrain) 
Ypredict_cart = cart_model_trained.predict(Xtest) 
Ypredict_id3 = id3_model_trained.predict(Xtest)

print("ID3 Tree Metrics")
print("Accuracy = ", met.accuracy_score(Ytest, Ypredict_id3))
print("Error Rate = ",1-met.accuracy_score(Ytest, Ypredict_id3))
print("Recall = ",met.recall_score(Ytest, Ypredict_id3,average ='weighted')) 
print("Precision = ",met.precision_score(Ytest,Ypredict_id3,average = 'weighted')) 
print("F-Measure = ",met.f1_score(Ytest, Ypredict_id3,average = 'weighted'))

print("CART Tree Metrics")
print("Accuracy = ", met.accuracy_score(Ytest, Ypredict_cart))
print("Error Rate = ",1-met.accuracy_score(Ytest, Ypredict_cart))
print("Recall = ",met.recall_score(Ytest, Ypredict_cart,average ='weighted'))
print("Precision = ",met.precision_score(Ytest,Ypredict_cart,average = 'weighted'))