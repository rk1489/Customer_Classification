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


import streamlit as st

def classify(num):
    if num == 0:
        return 'Class 1'
    elif num == 1:
        return 'Class 2'
    elif num == 2:
        return 'Class 3'
    elif num == 3:
        return 'Class 4'
    elif num == 4:
        return 'Class 5'
    elif num == 5:
        return 'Class 6'
    else:
        return 'Unidentified Class'
def main():
    st.title("Customer Classifier ML App")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['K - Nearest Neighbour (Accuracy = 97.5%)','Support Vector Classifier (Accuracy = 38.8%)','Decision Tree (Accuracy = 95.0%)', 'Random Forest (Accuracy = 96.3%)']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sex=["Male", "Female"]
    gender=sex.index(st.selectbox('Select Gender', sex))
    age=st.slider('Select Age', 18.0, 70.0)
    a_income=st.slider('Select Annual Income (k$)', 0.0, 150.0)
    s_score=st.slider('Select Spending Score', 1.0, 100.0)
    
    inputs=[[age,a_income,s_score,gender]]
    
    if st.button('Classify'):
        if option=='K - Nearest Neighbour (Accuracy = 97.5%)':
            st.success(classify(knn.predict(inputs)))
        elif option=='Support Vector Classifier (Accuracy = 38.8%)':
            st.success(classify(svm.predict(inputs)))
        elif option=='Decision Tree (Accuracy = 95.0%)':
            st.success(classify(decision_tree.predict(inputs)))
        elif option=='Random Forest (Accuracy = 96.3%)':
            st.success(classify(random_forest.predict(inputs)))
            

if __name__=='__main__':
    main()
