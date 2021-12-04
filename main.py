import pandas as pd
from sklearn.tree import DecisionTreeClassifier
a=pd.read_csv("music.csv")
print(a)
b=a.drop(columns=["genre"])
print(b)
c=a["genre"]
print(c)


model=DecisionTreeClassifier()
model.fit(b,c)
prediction=model.predict([ [28,1],[32,0] ])
print(prediction)