import streamlit as st
import pickle



knn_model=pickle.load(open('knn_model.pkl','rb'))
svm_model=pickle.load(open('svm_model.pkl','rb'))
tree_model=pickle.load(open('tree_model.pkl','rb'))
rf_model=pickle.load(open('rf_model.pkl','rb'))

def classify(num):
    if num < 0.5:
        return 'Class 1'
    elif num < 1.5:
        return 'Class 2'
    elif num < 2.5:
        return 'Class 3'
    elif num < 3.5:
        return 'Class 4'
    elif num < 4.5:
        return 'Class 5'
    elif num < 5.5:
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
            st.success(classify(knn_model.predict(inputs)))
        elif option=='Support Vector Classifier (Accuracy = 38.8%)':
            st.success(classify(svm_model.predict(inputs)))
        elif option=='Decision Tree (Accuracy = 95.0%)':
            st.success(classify(tree_model.predict(inputs)))
        elif option=='Random Forest (Accuracy = 96.3%)':
            st.success(classify(rf_model.predict(inputs)))
            

if __name__=='__main__':
    main()
