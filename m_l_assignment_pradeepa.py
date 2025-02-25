from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("weatherAUs.csv")

check_null_values = df["MinTemp"].isna().sum()
print(check_null_values)

df["MinTemp"] = df["MinTemp"].ffill()
print(df["MinTemp"])

null_values = df["Rainfall"].isna().sum()
print(null_values)

df["Rainfall"] = df["Rainfall"].ffill()
print(df["Rainfall"])

Le = LabelEncoder()

for columns in ['RainToday','RainTomorrow']:
    df[columns] = Le.fit_transform(df[columns])
    print(df[columns])

X =  df[["MinTemp","Rainfall"]]
Y =  df[["RainTomorrow"]]
Y = df[["RainToday"]]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.2,random_state=42)
print(X_test)
print(Y_test)
model = LogisticRegression ()
model.fit(X_train, Y_train)
Y_prediction = model.predict(X_test)
print(Y_prediction)

accuracy_check = accuracy_score(Y_test, Y_prediction)
print(accuracy_check)


