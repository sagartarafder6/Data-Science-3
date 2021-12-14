
'''
    Name: Prashant Kumar
    Roll No.: B19101
    Mobile Number: 8700350173

'''

import pandas as pd # importing all necessary Libraries 
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"landslide_data3.csv")   # Reading landslide_data3.csv file using panda
attributes = list(df.head(0)) # list of all attributes

#----------------------------Defining all necessary functions----------------------------------#
def outlier(data,attr): # funtion to calculate the outliner of an attribute 
    minimum = 2.5 * np.percentile(data[attr], 25) - 1.5 * np.percentile(data[attr], 75)
    maximum = 2.5 * np.percentile(data[attr], 75) - 1.5 * np.percentile(data[attr], 25)
    return pd.concat((data[attr][data[attr] < minimum], data[attr][data[attr] > maximum]))

def min_max_scaling(df,maxValue,minValue): # function to normalized data
    df_norm = df.copy()# copy the dataframe
    for column in attributes[2:]:  # normalizing data using min-max scaling
        min_df = df_norm[column].min()
        max_df = df_norm[column].max()
        df_norm[column] = (((df_norm[column] - min_df) / (max_df - min_df))*(maxValue-minValue))+minValue    
    return df_norm

def z_score_Normalization():# function to normalized data using z_score
    for column in attributes[2:]:  # normalizing data using z-score normalization
        df[column] = ((df[column] - df[column].mean()) / (df[column].std()))    
    return df    

def showingMinMax(df): # function for showing min and max value in tabular form
    dataMin=[]
    dataMax=[]
    for column in attributes[2:]: # iterarting over each column
        dataMin.append(df[column].min()) # finding min
        dataMax.append(df[column].max()) # finding max
    data = {"":attributes[2:],
            "Min":dataMin,"Max":dataMax}
    data = pd.DataFrame(data) # forming dataframe
    data = data.set_index([""])
    print(data)
    
def showingMeanStd(df): # function for showing mean and standard deviation value in tabular form
    dataMean=[]
    dataStd=[]
    for column in attributes[2:]: # iterarting over each column
        dataMean.append(df[column].mean()) # finding mean
        dataStd.append(df[column].std()) # finding standard deviation
    data = {"":attributes[2:],
            "Mean":dataMean,"Std":dataStd}
    data = pd.DataFrame(data) # forming dataframe
    data = data.set_index([""])
    print(data)
def MSE(mat1,mat2): # function to calculate mean square root
    return mean_squared_error(mat1,mat2)
    
def RMSE( data1, data2): # function to calculate root mean square error    
    Mse = MSE(data1, data2)
    Mse = (Mse**0.5) 
    return Mse



#---------------------------------------------QUESTION 1--------------------------------------------------#

def Q1():  
    for column in attributes[2:]: # iterarting over each column
        s = df.copy()
        s.drop(outlier(df,column).index,inplace=True)
        df[column].loc[outlier(df,column).index] = s[column].median() # replacing outliner with median                
    print("1(a) Min-Max Normalization")
    print("\tMinimum and maximum value before normalization")
    showingMinMax(df) 
    df_normalized = min_max_scaling(df,9,3) # calling min max scaling
    print("\tMinimum and maximum value After normalization")  
    showingMinMax(df_normalized)
    print("\n1(b) Z-score Nomalization")
    print("\tMean and Standard Deviation before normalization")
    showingMeanStd(df)
    df_z_score = z_score_Normalization() # calling z_score
    print("\tMean and Standard Deviation before normalization")
    showingMeanStd(df_z_score)

#---------------------------------------------QUESTION 2--------------------------------------------------#

def Q2():
    mean=[0,0] # mean matrix
    covariance = np.array([[5,10],[10,13]]) # covariance matrix
    samples = np.random.multivariate_normal(mean,covariance,1000,check_valid='ignore') # sample of 2*1000 matrix    
    plt.scatter(samples[:,0],samples[:,1],color='blue',alpha=0.6) # scatter plot of sample
    print("2(a) Scatter plt of 1000 sample")
    plt.title("Scatter plot of 1000 samples")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show() # show plot
    vals,vect = np.linalg.eig(np.cov(samples.transpose())) # finding eigen values and eigen vector
    print("\n2(b) Eigenvalues And EigenVectors of covariance matrix")
    print("Eigen Values are: ",*vals)
    print("Eigen Vector are: ",*vect)
    proj = np.dot(samples,vect) # matrix of projection of samples on eigen vector
    one_x = [];one_y=[];second_x=[];second_y=[] # list for computing x and y coordinate for each projection
    for i in range(1000):# computing x and y coordinate for each projection
        s = proj[:,0][i]*vect[:,0]
        t = proj[:,1][i]*vect[:,1]
        one_x.append(s[0]);one_y.append(s[1])
        second_x.append(t[0]);second_y.append(t[1])
    plt.figure(figsize=(7,7)) 
    plt.scatter(samples[:,0],samples[:,1],color='blue',alpha=0.6)  # scatter plot of sample
    plt.quiver(0,0,vect[0][0],vect[1][0],scale=6.5,color='red',angles="xy") # ploting first eigen value
    plt.quiver(0,0,vect[0][1],vect[1][1],scale=2.3,color='red',angles="xy")# ploting second eigen value
    plt.title("Plot of 2D synthetic data and eigen directions") 
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.axis("equal")
    plt.show()
    
    print("\n2(c) Projecting data on eigen vectors")
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    axs[0].scatter(samples[:,0],samples[:,1],color='blue',alpha=0.6)  # scatter plot of sample
    axs[0].quiver(0,0,vect[0][0],vect[1][0],scale=6.5,color='red',angles="xy",alpha=1) # ploting first eigen value
    axs[0].quiver(0,0,vect[0][1],vect[1][1],scale=2.3,color='red',angles="xy",alpha=1) # ploting second eigen value
    axs[0].scatter(one_x,one_y,color='#f984dd',alpha=0.4,marker="*") # projection on one eigen vector
    axs[0].set_xlabel("X1")
    axs[0].set_ylabel("X2")
    axs[0].set_title("Projected values onto the first eigen directions")
    axs[1].scatter(samples[:,0],samples[:,1],color='blue',alpha=0.6,)  # scatter plot of sample
    axs[1].quiver(0,0,vect[0][0],vect[1][0],scale=6.5,color='red',angles="xy")# ploting first eigen value
    axs[1].quiver(0,0,vect[0][1],vect[1][1],scale=2.3,color='red',angles="xy")# ploting second eigen value
    axs[1].scatter(second_x,second_y,color='#f984dd',marker="*")# projection on another eigen vector
    axs[1].set_xlabel("X1")
    axs[1].set_ylabel("X2")
    axs[1].set_title("Projected values onto the second eigen directions")
    plt.show()
        
    newD = np.dot(vect,proj.transpose()) # reconstruct the samples data
    print(newD,samples)
    print("\n2(d) Mean Square Error between original Data and reconstructed Data")
    mse=MSE(newD,samples.transpose()) # finding mean square error
    print("\tMean Square error: "+str(mse)) 



def Q3():
    pca = PCA(n_components=2).fit_transform(df.iloc[:,2:]) # compressed data       
    eigenValue = np.linalg.eigvals(np.cov(pca.transpose())) # finding eigen value of compressed data
    print(np.cov(pca.transpose()))
    print("\n3(a) Variance and eigenvalue\n ")
    print("\tVariance of Dimension [0]: ",np.var(pca[:, 0]))
    print("\tEigen VAlue of Dimension [0]:", eigenValue[0])
    print("\tVariance oft Dimension [1]: ",np.var(pca[:, 1]))
    print("\tEigen VAlue of Dimension [1]:", eigenValue[1])
    
    plt.scatter(pca[:,0],pca[:,1],color='red',marker='^') # scatter plot of compressed data
    plt.title("Plot of two dimensional of reduced data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    
    eigenValue = np.linalg.eigvals(np.cov(df.iloc[:,2:].transpose())).tolist() # eigen values of z_score data
    eigenValue.sort(reverse=True) # sort the array in descending order
    print("\n3(b) Plotting Eigen values in descending order")
    plt.plot(range(1,len(eigenValue)+1), eigenValue,marker="o",linestyle="--",color='orange') # ploting eigen values
    plt.yscale("log") # changing y-scale to logarithmic
    plt.title("Plot of eigenvalues in descending order")
    plt.xlabel("x-axis")
    plt.ylabel("Value of Eigenvalue")
    plt.show() # show plot
    
    orig_data = df.iloc[:,2:] # original data
    Rmse_List=[] # list of rmse 
    for i in range(1,len(orig_data.columns)+1):
        pca = PCA(n_components=i) 
        compData = pca.fit_transform(orig_data) # forming data into ith dimension
        reConstData = pca.inverse_transform(compData) # reconstructing the data 
        Rmse_List.append(((RMSE(orig_data,reConstData)))) # calling function to calculate rmse
    
    print("\n3(c) Plot of RMSE v/s L for each attribute")
    plt.plot(range(1,len(orig_data.columns) + 1),Rmse_List,marker="o",linestyle="--",color='green') # ploting bar graph of rmse v/s l
    plt.ylabel("RMSE")
    plt.title("Plot of RMSE v/s L for each attribute")
    plt.xlabel("X-axis = l")
    plt.show() # showing plot
print("-------------------------------------------ANSWER 1--------------------------------------------------")
Q1()
print()
print("-------------------------------------------ANSWER 2--------------------------------------------------")
Q2()
print()
print("-------------------------------------------ANSWER 3--------------------------------------------------")
Q3()