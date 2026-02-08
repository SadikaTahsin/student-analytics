import pandas as pd

data = pd.read_csv("student_data.csv")

print(data.head())

# Encode categorical
data['Gender'] = data['Gender'].map({'Male':0,'Female':1})
data['InternetAccess'] = data['InternetAccess'].map({'No':0,'Yes':1})

print(data.head())

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(data.corr(), annot=True)
plt.show()

sns.scatterplot(x=data['StudyHours'], y=data['FinalScore'])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X = data.drop('FinalScore', axis=1)
y = data['FinalScore']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test,pred))

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.plot(kind='barh')
plt.show()

import streamlit as st
import pandas as pd

data = pd.read_csv("student_data.csv")

st.title("🎓 Student Performance Dashboard")

st.dataframe(data)

st.bar_chart(data['FinalScore'])

hours = st.slider("Study Hours",1,10,5)
st.write("Selected Study Hours:", hours)

print("\n📊 KEY INSIGHTS:")

print("Average Score:", data['FinalScore'].mean())

print("Top Study Hours Impact:")
print(data.groupby('StudyHours')['FinalScore'].mean())

print("Attendance Impact:")
print(data.groupby('Attendance')['FinalScore'].mean())

import joblib

joblib.dump(model, "student_model.pkl")
print("Model saved!")

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_model.pkl")

st.title("🎓 Student Score Predictor")

gender = st.selectbox("Gender", ["Male","Female"])
study = st.slider("Study Hours",1,10,5)
attendance = st.slider("Attendance %",50,100,80)
internet = st.selectbox("Internet Access",["Yes","No"])
previous = st.slider("Previous Score",40,100,70)
age = st.slider("Age",15,20,17)

# Encode
gender = 0 if gender=="Male" else 1
internet = 1 if internet=="Yes" else 0

input_data = [[gender,age,study,attendance,internet,previous]]

if st.button("Predict Score"):
    pred = model.predict(input_data)
    st.success(f"Predicted Final Score: {pred[0]:.2f}")
