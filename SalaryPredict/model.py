import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

Salary=pd.read_csv('/Users/khadeejah/Desktop/datasets-9401-13260-Salary_Data.csv')
Salary.head()

Salary.isnull().sum()
X=Salary.iloc[:,:-1].values.reshape(-1,1)
y=Salary.iloc[:,-1].values.reshape(-1,1)

#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
rg=LinearRegression()
rg.fit(X,y)

y_pred=rg.predict(X_test)
y_pred

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
#savin model
pickle.dump(rg,open('model.pkl','wb'))
model= pickle.load(open('model.pkl','rb'))
model.predict([[4]])