import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("student_scores.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
st.title("exam score predictor")
st.write("enter hours studied to predict the exam scores.")
hours = st.number_input("hours studied:",min_value=0.0, step=0.1)
if st.button("predict scores"):
      predicted_scores = model.predict([hours])[0]
      st.success(f"predicted scores:  {predicted_scores:.2f}")
st.write("###   Sample Training Data")
st.dataframe(df)
