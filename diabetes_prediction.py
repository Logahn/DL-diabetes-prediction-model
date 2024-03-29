import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes=pd.read_csv('diabetes.csv')
print(diabetes.columns)

diabetes.head()

print("dimension  of the data: {}".format(diabetes.shape))

#grouping data based on outcome
print(diabetes.groupby('Outcome').size())

import seaborn as sns
sns.countplot(diabetes['Outcome'], label="Count");

diabetes.info()

"""### K-Nearest Neighbours to Predict Diabetes"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(diabetes.loc[:,diabetes.columns!='Outcome'], diabetes['Outcome'], random_state=80, train_size=.6)

from sklearn.neighbors import KNeighborsClassifier

train_accuracy=[]
test_accuracy=[]

nbd=range(1,15)

for n_nbd in nbd:
    #build the model
    knn=KNeighborsClassifier(n_neighbors=n_nbd)
    knn.fit(x_train, y_train)
    
    #record the accuracy 
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))
    
plt.plot(nbd, train_accuracy, label="train accuracy")
plt.plot(nbd, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("nbd")

plt.legend()

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
print(knn.score(x_train, y_train))
print(knn.score(x_test, y_test))

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(x_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(x_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg001.score(x_train, y_train)))
# print("Test set accuracy: {:.3f}".format(logreg001.score(x_train, y_test)))

logreg100 = LogisticRegression(C=100).fit(x_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg100.score(x_train, y_train)))
# print("Test set accuracy: {:.3f}".format(logreg100.score(x_train, y_test)))

diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=9]
plt.figure(figsize=(8,6))
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
plt.hlines(0, 0, diabetes.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('log_coef')

"""### Decision Tree Classifier to predict diabetes"""

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(random_state=0)

tree.fit(x_train, y_train)
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))

tree=DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x_train, y_train)

print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))

"""### Feature Importance in Decision Trees"""

print(tree.feature_importances_)

diabetes_features=diabetes.loc[:,diabetes.columns!='Outcome']
def plot_FI(model):
    plt.figure(figsize=(8,6))
    features=8
    plt.barh(range(features),model.feature_importances_)
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.yticks(np.arange(features), diabetes_features)
    
    plt.ylim(-1,features)

plot_FI(tree)

"""### Deep Learning to Predict Diabetes"""

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=45)

mlp.fit(x_train, y_train)

print(mlp.score(x_train, y_train))
print(mlp.score(x_test, y_test))

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_scale=scale.fit_transform(x_train)
x_test_scale=scale.fit_transform(x_test)

mlp=MLPClassifier(random_state=0)

mlp.fit(x_train_scale, y_train)

print(mlp.score(x_train_scale, y_train))
print(mlp.score(x_test_scale, y_test))

plt.figure(figsize=(20,20))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')

plt.yticks(range(8),diabetes_features)
plt.xlabel("Weight matrix")
plt.ylabel("Features")

plt.colorbar()
