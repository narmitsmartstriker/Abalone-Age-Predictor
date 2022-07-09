#B20218
#Narmit Kumar
#Ph : 8580496363

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures

a = pd.read_csv("abalone.csv")

X = a.iloc[:,:-1]
Y = a["Rings"]


# Splitting data into training and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state=42,shuffle=True)
train = pd.concat([X_train,Y_train],axis = 1)
test = pd.concat([X_test,Y_test],axis = 1)

#Creating csv files for train and test data
train.to_csv("abalone_train.csv" , index = False)
test.to_csv("abalone_test.csv" , index = False)

#Q1.
print("Q1:",'\n')

#Finding attribute having highest correlation with Target attribute "Rings".
cr =a.corr()
input_var = cr.iloc[:-1,-1].idxmax()

#Building Linear Regression Model to predict 'Rings'
reg = LinearRegression().fit(np.array(train["Shell weight"]).reshape(-1, 1), train['Rings'])

#A part : plotting best fit line
x = np.linspace(0, 1, 2923).reshape(-1, 1)
y = reg.predict(x)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(x, y, color='gold')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Line')
plt.show()

#B part : prediction accuracy on the training data using root mean squared error
print('b:')
y_train_pred = reg.predict(np.array(train["Shell weight"]).reshape(-1, 1))
rmse_train = (MSE(train['Rings'], y_train_pred)) ** 0.5
print("The rmse for training data is", round(rmse_train, 3))

#C part : prediction accuracy on the test data using root mean squared error
print('c:')
y_test_pred = reg.predict(np.array(test["Shell weight"]).reshape(-1, 1))
rmse_test = (MSE(test['Rings'].to_numpy(), y_test_pred)) ** 0.5
print("The rmse for testing data is", round(rmse_test, 3))
print('\n')

#D part:Plotting the scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data
plt.scatter(test['Rings'], y_test_pred)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Simple linear regression model')
plt.show()


#Q2.multivariate (multiple) linear regression model to predict Rings
print("Q2",'\n')

#A :prediction accuracy on the training data using root mean squared error
print('a:')
reg_train = LinearRegression().fit(X_train, Y_train)
rmse_train = (MSE(Y_train, reg_train.predict(X_train))) ** 0.5
print("The rmse for training data is", round(rmse_train, 3))

#B : prediction accuracy on the test data using root mean squared error
print('b:')
reg_test = LinearRegression().fit(X_test, Y_test)
rmse_test = (MSE(Y_test, reg_test.predict(X_test))) ** 0.5
print("The rmse for testing data is", round(rmse_test, 3))
print('\n')

#C: scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data
plt.scatter(Y_test, reg_test.predict(X_test))
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Multivariate linear regression model')
plt.show()

#Q3.A simple nonlinear regression model using polynomial curve fitting to predict Rings
print("Q3:")
P = [2, 3, 4, 5]

#A
print('a:')
X = np.array(train['Shell weight']).reshape(-1, 1)
RMSE = []
for i in P:
    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    y_pred = reg.predict(x_poly)
    rmse = (MSE(Y_train, y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", i, 'is', round(rmse, 3))

#Plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.ylim(2.3,2.55)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Univariate non-linear regression model")
plt.show()

#B
print('b:')
RMSE = []

X = np.array(test['Shell weight']).reshape(-1, 1)
Y_pred = []
for i in P:
    poly_features = PolynomialFeatures(i)  
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (MSE(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", i, 'is', round(rmse, 3))
    

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.ylim(2.25,2.55)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE (test data)')
plt.title("Univariate non-linear regression model")
plt.show()

#C
#Taking p = 5 which is having lowest RMSE
x_poly = PolynomialFeatures(5).fit_transform(x)
reg = LinearRegression()
reg.fit(x_poly, Y_train)
y1 = reg.predict(x_poly)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(np.linspace(0, 1, 2923), y1 , color='gold')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Curve')
plt.show()

#D
#The best degree of polynomial is 5.
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate non-linear regression model')
plt.show()

#Q4.Multivariate nonlinear regression model using polynomial regression to predict Rings
print("Q4:")

#A
print('a:')
RMSE = []
for i in P:
    poly_features = PolynomialFeatures(i)  
    x_poly = poly_features.fit_transform(X_train)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (MSE(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", i, 'is', round(rmse, 3))

#Plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Multivariate non-linear regression model")
plt.show()

#B
print('b:')
RMSE = []
Y_pred = []
for i in P:
    poly_features = PolynomialFeatures(i)  # p is the degree
    x_poly = poly_features.fit_transform(X_test)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (MSE(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", i, 'is', round(rmse, 3))
    #C
    #The best degree of polynomial is 3 as p=3 has minimum rmse
    if i == 3:
        plt.scatter(Y_test, Y_pred)
        plt.xlabel('Actual Rings')
        plt.ylabel('Predicted Rings')
        plt.title('Univariate non-linear regression model')
        plt.show()

#Plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Multivariate non-linear regression model")
plt.show()

