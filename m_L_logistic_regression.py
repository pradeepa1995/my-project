from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

X = df[["MinTemp","Rainfall"]]
Y_RainTomorrow = df["RainTomorrow"]
Y_RainToday = df["RainToday"]

# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_RainTomorrow, test_size=0.2, random_state=42)

model = LogisticRegression ()
model.fit(X_train, Y_train)
y_prediction = model.predict(X_test)
print(y_prediction)

accuracy_check = accuracy_score(Y_test, y_prediction)
print(accuracy_check)

# Now, predict RainToday
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_RainToday, test_size=0.2, random_state=42)

model = LogisticRegression ()
model.fit(X_train, Y_train)
y_prediction = model.predict(X_test)
print(y_prediction)

accuracy_check = accuracy_score(Y_test, y_prediction)
print(accuracy_check)


