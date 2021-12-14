'''
    Name: Prashant Kumar
    Roll No.: B19101
    Mobile Number: 8700350173

'''
#----------------------------------Import Libraries---------------------------------#
import pandas as pd # importing all necessary Libraries 
import numpy as np
import math as m
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 

#----------------------------------Reading file-------------------------------------#

train_data = pd.read_csv(r"seismic-bumps-train.csv").groupby("class") # Reading seismic-bumps-train.csv
test_data = pd.read_csv(r"seismic-bumps-test.csv") # Reading seismic-bumps-test.csv

atm_data = pd.read_csv(r"atmosphere_data.csv")  # Reading atmosphere_data.csv file using panda
[X_train, X_test, X_label_train, X_label_test]=train_test_split(atm_data.copy(), # spliting data into 70% train and 30% test
        atm_data['temperature'],test_size=0.3, random_state=42,shuffle=True)
X_train.to_csv('atmosphere-train.csv',index=False) # saving tarin data
X_test.to_csv('atmosphere-test.csv',index=False) # saving test data

plt.style.use('seaborn-whitegrid') # using seaborn-style to plot all graph

#-----------------------------------Defining all necessary functions------------------------#

def MSE(mat1,mat2): # define MSE function to calculate mean square error
    return mean_squared_error(mat1,mat2)

def Bayes_GMM(Q): # function to predict class on the basis of bayes on GMM
    print("--> For Q =",Q)   
    prior_C0 = len(train_data.get_group(0))/(len(train_data.get_group(0))+len(train_data.get_group(1))) # class 0 prior
    prior_C1=1-prior_C0 # class 1 prior
    gmm0 = GaussianMixture(n_components = Q, random_state=42)
    gmm0.fit(train_data.get_group(0).iloc[:,:-1]) # fitting class 0 of train data
    score_0 = gmm0.score_samples(test_data.iloc[:,:-1]) # log lokelihood probability of class 0

    gmm1 = GaussianMixture(n_components = Q, random_state=42)
    gmm1.fit(train_data.get_group(1).iloc[:,:-1]) # fitting class 1 of train data
    score_1 = gmm1.score_samples(test_data.iloc[:,:-1]) # log lokelihood probability of class 1  
    predict = [] # list containg prediction of class
    for i,j in zip(score_0,score_1):
        predict.append(0 if m.exp(i)*prior_C0>m.exp(j)*prior_C1 else 1) # comparing posterior probaility to assigning class

    accuracy = accuracy_score(test_data['class'],predict) # sklearn funct. to calculate accuracy
    conf_mat = confusion_matrix(test_data['class'],predict) # sklearn function to find confusion matrix
    print("Confusion Matrix: ")
    print(conf_mat)
    print("Accuracy:",accuracy)
    return accuracy  # return accuracy 

def polynomial_curve_fit(p,train_data,train_label,test_data): #function to predict class using polynomial curve fitting 
    polynomial_features = PolynomialFeatures(p)  # using sklearn function
    x_poly = polynomial_features.fit_transform(train_data) # fitting train data
    regressor = LinearRegression()
    regressor.fit(x_poly,train_label)
    test_data = polynomial_features.fit_transform(test_data)
    predict = regressor.predict(test_data) # predicting test data
    return predict # returning prediction
    
def Show_Rmse_plot(train_data,train_label,test_data,mse_label,color): # function for plotting RMSE
    Rmse = [] # list containg RMSE for different value of p
    min_Rmse = 100
    best_fit = 2
    for i in range(2,6):
        a = polynomial_curve_fit(i,train_data,train_label,test_data) # calling polynomial curev fit function 
        mse = MSE(a,mse_label) # calculating MSE
        Rmse.append(mse**0.5) # appending RMSE 
        if min_Rmse > Rmse[-1]: # comparing to find minimum accuracy
            min_Rmse = Rmse[-1] 
            best_fit = i # finding best fit that has less RMSE
    
    for i in range(2,6): # loop for printing RMSe for each p
        print("* P = "+str(i)+" is "+str(Rmse[i-2]))
    plt.bar(range(2,6),Rmse,color=color,align='center',width=[0.35]*len(range(2,6))) # plotting bar graph of RMSe for different p
    plt.xticks(range(2,6),range(2,6))
    plt.xlabel('Degree of Polynomial (p)')
    plt.ylabel('RMSE')
    plt.show()
    return best_fit # retutrning p that is good fit

#-----------------------------------Solution of PartA---------------------------------#
def PartA():
    Q = [2,4,8,16] # list of Q
    max_Q_accu = 0
    max_accu = 0
    for i in Q:
        accu = Bayes_GMM(i) # calling BAyes_GMM function
        if accu > max_accu: # comparing to fing max accuracy
            max_accu = accu
            max_Q_accu = i
    print(f"\nHigh Accuracy is for Q = {max_Q_accu} is {max_accu}")

#-----------------------------------Solution of PartB_Q1---------------------------------#    
def PartB_Q1():
    print(" --> 1.(a)")
    regressor = LinearRegression() # using sklearn function
    x_train = X_train.iloc[:,1].values.reshape(-1,1)
    x_test = X_test.iloc[:,1].values.reshape(-1,1)
    regressor.fit(x_train,X_label_train) # fitting train data
    predict = regressor.predict(x_test) # predicting value of test data
    plt.scatter(X_train.iloc[:,1],X_label_train,color="blue")   # plotting scatter plot     
    plt.plot(X_train.iloc[:,1],regressor.predict(x_train) ,color='red') # plotting best fit line
    plt.title("Best Fit Line on Training Data")
    plt.xlabel('Pressure')
    plt.ylabel('Temperature')
    plt.show()
                
    print("\n --> 1.(b)\n Prediction Accuracy on Training Data using RMSE: ",MSE(X_label_train,regressor.predict(x_train))**0.5) # calling MSE function to calculate RMSE
    print("\n --> 1.(c)\n Prediction Accuracy on Test Data using RMSE: ",(MSE(predict,X_label_test))**0.5)  # calling MSE function to calculate RMSE
    
    print("\n --> 1.(d)")
    plt.scatter(X_label_test,predict,color = "#d11554") # ploting scatter plot
    plt.title("Actual Temp. v/s Predicted Temp.")
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.axis('equal')
    plt.show()

#-----------------------------------Solution of PartB_Q2---------------------------------#
def PartB_Q2():
    print(" --> 2.(a)\nPrediction accuracy on Training Data using RMSE for:")
    plt.title("Prediction Accuracy on Training Data using RMSE")
    train_fit = Show_Rmse_plot(X_train.iloc[:,1].values.reshape(-1,1),X_label_train, # calling show_RMSE_plot function
             X_train.iloc[:,1].values.reshape(-1,1),X_label_train,'red')
    print("\n --> 2.(b)\nPrediction accuracy on Test Data using RMSE for:")
    plt.title("Prediction Accuracy on Test Data using RMSE")
    best_fit = Show_Rmse_plot(X_train.iloc[:,1].values.reshape(-1,1),X_label_train, # calling show_RMSE_plot function
             X_test.iloc[:,1].values.reshape(-1,1),X_label_test,'#16c918')
    print("\n --> 2.(c)")
    test_pred = polynomial_curve_fit(best_fit,X_train.iloc[:,1].values.reshape(-1,1),X_label_train,X_test.iloc[:,1].values.reshape(-1,1)) # calling polynomial curve fit to find prediction
    p = np.polyfit(X_test.iloc[:,1],test_pred,best_fit) # function to compute lest squares polynomial
    X_seq = np.linspace(X_train.iloc[:,1].min(),X_train.iloc[:,1].max(),500).reshape(-1,1) # function to get number of points over a particular range
    plt.figure()
    plt.plot(X_seq,np.polyval(p,X_seq),color='red')# plotting best fit curve 
    plt.scatter(X_train.iloc[:,1],X_label_train,color="#ac1bc1") # plotting scatter plot
    plt.title("Best Fit Curve on Training Data")
    plt.xlabel('Pressure')
    plt.ylabel('Temperature')
    plt.show()
    print("\n --> 2.(d)")
    plt.scatter(X_label_test,test_pred,color="blue") # plotting scatter plot
    plt.title("Actual Temp. v/s Predicted Temp.")
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.axis('equal')
    plt.show()

print("\n---------------------------------------PART A--------------------------------------")    
PartA()    
print("\n---------------------------------------PART B--------------------------------------")    
print('>----------ANSWER 1----------<')
PartB_Q1()
print('\n>----------ANSWER 2----------<')
PartB_Q2()