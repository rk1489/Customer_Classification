import streamlit as st
import pickle



knn=pickle.load(open('knn.pkl','rb'))
tree=pickle.load(open('tree.pkl','rb'))
rf=pickle.load(open('rf.pkl','rb'))

def classify(n):
    if n == 'Class 1':
        return 'Class 1 (Young middle responsible behaviour)'
    elif n == 'Class 2':
        return 'Class 2 (Old middle responsible behaviour)'
    elif n == 'Class 3':
        return 'Class 3 (Aged rich miser behaviour)'
    elif n == 'Class 4':
        return 'Class 4 (Young rich spendthrift behaviour)'
    elif n == 'Class 5':
        return 'Class 5 (Young poor spendthrift behaviour)'
    elif n == 'Class 6':
        return 'Class 6 (Aged poor responsible behaviour)'
    else:
        return 'Unidentified Customer Class'
    
def main():
    st.title("Customer Classifier ML App")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['K - Nearest Neighbour (Accuracy = 97.5%)','Decision Tree (Accuracy = 95.0%)', 'Random Forest (Accuracy = 96.3%)']
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
        elif option=='Decision Tree (Accuracy = 95.0%)':
            st.success(classify(tree.predict(inputs)))
        elif option=='Random Forest (Accuracy = 96.3%)':
            st.success(classify(rf.predict(inputs)))
            

if __name__=='__main__':
    main()
