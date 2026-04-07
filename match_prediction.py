import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("football_dataset.xlsx")

df['Result'] = df['Result'].map({'Win':1,'Loss':0})

X = df[['Goals','Shots','Possession (%)']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("Accuracy:", accuracy)
