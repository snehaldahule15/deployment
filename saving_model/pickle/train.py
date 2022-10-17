import array
from copyreg import pickle
from fileinput import filename
from unittest import result
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle #save model librery

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg','plas','pres','skin','test','mass','predi','age','class']
dataframe = pd.read_csv(url, names=names)
print(dataframe)

#seperate independent and target variable.
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#train test split
x_train, x_test, y_train, y_test =model_selection.train_test_split(x, y, test_size=0.2, random_state=101)

#fir the model
model = LogisticRegression()
model.fit(x_train,y_train)

#accuracy
result = model.score(x_test, y_test)
print(result)


#saving the model
filename = "diabetes_79.pkl"
pickle.dump(model, open(filename, 'wb'))#wb==byte mode















