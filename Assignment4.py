'''
#Question 1:

#Libraries required
import pandas as pd
import matplotlib.pyplot as plt
#Reading the csv file into dataframe
data = pd.read_csv("C:/Users/sruja/OneDrive/Desktop/Neural Network/Neural-git/Assignments/data.csv")
#Printing top few rows
print(data.head())
data.describe()
#Replacing null values found with mean
data = data.fillna(data.mean())
#Select at least two columns and aggregate the data using: min, max, count, mean
print(data[['Maxpulse','Calories']].agg(['min','max','mean','count']))
#Filter the dataframe to select the rows with calories values between 500 and 1000.
print(data[(data.Calories < 1000) & (data.Calories > 500)])
#Filter the dataframe to select the rows with calories values > 500 and pulse < 100.
print(data[ (data.Calories>500) & (data.Pulse < 100)])
#Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”.
df_modified = pd.DataFrame(data, columns = ['Duration', 'Pulse', 'Calories'])
print (df_modified.head())
#Delete the “Maxpulse” column from the main df dataframe
del data["Maxpulse"]
print(data.head)
#Convert the datatype of Calories column to int datatype.
data['Calories'] = data['Calories'].astype("int")
print(data['Calories'].dtypes)
#Using pandas create a scatter plot for the two columns (Duration and Calories).
print(data.plot.scatter(x ='Duration', y= 'Calories'))
plt.show()


'''

#Question 2:

#Libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Import the given “Salary_Data.csv” 
train_test = pd.read_csv('C:/Users/sruja/OneDrive/Desktop/Neural Network/Neural-git/Assignments/Salary_Data.csv')
#Dividing the dataframes
X = train_test.iloc[:, 0:1].values
y = train_test.iloc[:, 1].values
# Dividing data into two dataframes in a way that test datea is 1/3rd of original data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)
# Performing linear regression on train dataframe
model = LinearRegression()
model.fit(X_train, y_train)
# Predicting the Test set results
y_pred = model.predict(X_test)
print(y_pred)
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'yellow')
plt.plot(X_train, model.predict(X_train), color = 'green')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Total years of experience')
plt.ylabel('Salary')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'yellow')
plt.plot(X_train, model.predict(X_train), color = 'green')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Total years of experience')
plt.ylabel('Salary')
plt.show()


