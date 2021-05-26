import pandas as pd

df = pd.read_csv('Customer_Class.csv')

# Defining data and label
X = df.iloc[:, 1:5]
y = df.iloc[:, 5]

# Spliting data into training and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Applying KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
knn.fit(X_train, y_train)

#Applying SVC algorithm
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, y_train)

# Applying Decision Tree
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(criterion='gini')
decision_tree.fit(X_train, y_train)

# Using Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Importing pickle for deployment
import pickle

pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(svm,open('svm_model.pkl','wb'))
pickle.dump(decision_tree,open('tree_model.pkl','wb'))
pickle.dump(random_forest,open('rf_model.pkl','wb'))
