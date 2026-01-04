import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Title
st.title("Burnout Prediction System")

# Training data
data = {
    "Mood":   [5,4,3,2,1,2,3,5, 1,2,4,5],
    "Sleep":  [8,7,6,5,4,4,5,8, 3,4,7,8],
    "Screen": [4,5,6,7,9,8,7,3, 10,9,4,3],
    "Work":   [5,6,7,8,10,9,8,4, 11,9,6,4],
    "Burnout":["Low","Low","Medium","Medium","High","High","Medium","Low",
                "High","High","Low","Low"]
}


df = pd.DataFrame(data)

le = LabelEncoder()
df["BurnoutEncoded"] = le.fit_transform(df["Burnout"])

X = df[["Mood","Sleep","Screen","Work"]]
y = df["BurnoutEncoded"]

model = LogisticRegression()
model.fit(X, y)

# User inputs
mood = st.slider("Mood (1 = bad, 5 = good)", 1, 5, 3)
sleep = st.slider("Sleep hours", 0, 12, 6)
screen = st.slider("Screen time (hours)", 0, 15, 6)
work = st.slider("Work hours", 0, 15, 8)

if st.button("Predict Burnout"):
    user_input = pd.DataFrame(
        [[mood, sleep, screen, work]],
        columns=["Mood","Sleep","Screen","Work"]
    )
    pred = model.predict(user_input)
    result = le.inverse_transform(pred)[0]

    st.success(f"Predicted Burnout Level: {result}")
