import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


st.subheader("아이리스 특성값 입력")

iris = load_iris()
X = iris.data
y = iris.target

target_names = iris.target_names

X_train,X_test,y_train,y_test = train_test_split(X,y)







knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

c1,c2,c3,c4 = st.columns(4)
with c1:
    sepal_length = st.number_input("sepal length",value=5.8, step=0.1)
with c2:    
    sepal_width = st.number_input("sepal width",value=3.1, step=0.1)

with c3:
    petal_length = st.number_input("petal_length",value=3.8, step=0.1)
with c4:    
    petal_width = st.number_input("petal length",value=1.2, step=0.1)


if st.button("예측하기"):
    input_x = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    pred = knn.predict(input_x)[0]
    st.success(f'예측 꽃 이름 : {pred} /  {target_names[pred]} 입니다. ❤❤❤')