# -*- coding: utf-8 -*-

"""
Created on Tue Jul 25 07:25:47 2023

@author: domingosdeeulariadumba
"""

%pwd



""" Importing the required libraries """


# For EDA and Plotting

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mpl
mpl.style.use('ggplot')


# For ML procedures

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as SScl, normalize


# To summarize the estimation model

import statsmodels.api as sm


# To ignore warnings about compactibility issues and so on

import warnings
warnings.filterwarnings('ignore')



"""" EXPLORATORY DATA ANALYSIS """


# Loading the dataset and extracting the relevant columns

    '''
    Importing the dataset.
    '''
dataset=pd.read_excel("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/PrivateBankingBigFiveAngola.xlsx", sheet_name=None)

    '''
    Concatenating the excel sheets.
    '''
df=pd.concat(dataset.values(), axis=1)

    '''
    Extracting the relevant columns
    '''
df=df[['Representation', 'Training Hours', 'Until Secondary School', 'Degree Level', 'Bachelor/College Attendance', 'Others', 'Net Income (mMAOA)']]

    '''
    Checking the Data type
    '''
df.dtypes


# Renaming the 'Representation' column.

df=df.rename({'Representation':'Most Expressive Age Range Concentration'}, axis=1)


# Rounding the dataframe values

df=df.round(2)


# Presenting the statisctical summary

df.describe()


# Displaying the KDE plots

var= ['Most Expressive Age Range Concentration',
       'Training Hours',
       'Until Secondary School', 'Until Secondary School', 'Degree Level',
       'Bachelor/College Attendance', 'Others', 'Net Income (mMAOA)']

for i in var:
    mpl.figure()
    sb.kdeplot(df[i])
    mpl.savefig('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/{0}_kdeplot.png'.format(i))
    mpl.close()    

    
# Showing the Pairplot

sb.pairplot(df)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/BankAo_PairPlot.png")
mpl.close()


# Visualizing the correlation heatmap

sb.heatmap(df.corr(), annot=True, cmap='coolwarm')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/BankAo_CorrelationHeatmap.png")
mpl.close()


# Combining each independent variables with the dependent variable in pairplot

sb.pairplot(df,x_vars=['Most Expressive Age Range Concentration',
       'Training Hours',
       'Until Secondary School', 'Until Secondary School', 'Degree Level',
       'Bachelor/College Attendance', 'Others'], y_vars=['Net Income (mMAOA)'],
                height=5, aspect=.8, kind="reg")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/BankAo_Pairplot2.png")
mpl.close()
 

""" DATA PREPROCESSING """

# Setting up the independent and dependent variables

X=df.iloc[:,:-1]


y=df.iloc[:,-1]



# Normalization, scaling and dimensionality reduction of the preditors set

    '''
    This process may be automatized bay using the function below
    '''
    
def Normalize_Scale_and_Reduce(X):
    
        
    # Normalizing the predictors due to 'Training Hours' having a different range
    
    norm_pred=normalize(X)
    
    
    # Scaling the normalized data to bring them to a comparable level
    
    scl_pred=SScl().fit_transform(norm_pred)
    
    
    # Reducing the dimensionality of the scaled data to avoid multicollinearity
                
         # Finding the ideal number of components to consider
                        
    pc=PCA(n_components=X.shape[1]).fit(scl_pred)
    pc_var=pc.explained_variance_ratio_
    pc_var_cumsum=np.cumsum(np.round(pc_var, decimals=4)*100)
    print('The ideal number of components is', np.unique(pc_var_cumsum).size)
    mpl.plot(pc_var_cumsum)
    mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/BankAo_Components.png")
    mpl.close()
                     
    new_pc=PCA(n_components=np.unique(pc_var_cumsum).size).fit_transform(scl_pred)
   
    
    return pd.DataFrame(new_pc)


# Creating new variables after normalization, scale and dimensionality reduction

x=Normalize_Scale_and_Reduce(X)



# Outliers treatment

mpl.boxplot(x)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/BankAo_PredictorsBxPlt.png")
mpl.close()

mpl.boxplot(y)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolanBig5BankingAnalysis/BankAo_TargetBxPlt.png")
mpl.close()

    """
    As shown above, there's ouliers just in the predictors. These'll be
    handled by using the imputation technique, considering the median value
    since the mean tends to be significantly influenced by outliers.
    """

def imputation (data):
        
    n_outliers=[]
    q1=np.quantile(data, 0.25)
    q3=np.quantile(data, 0.75)
    iqr=q3-q1       
    
    for k in data:
        
        # There's only values beyond the upper whisker. Therefore, the
        # imputation will be done only for this area.
        
        if k>(q3+1.5*iqr):
            n_outliers.append(k)            
       
    data_imp = np.where(data>=min(n_outliers), data.median(), data)          
    
    return pd.DataFrame(data_imp)
        

        """
        Applying the imputation function
        """
for k in range(5):
    x[k]=imputation(x[k])



""" BUILDING THE ESTIMATION MODEL """


# Splitting the data into train and test sets

x_train, x_test, y_train, y_test = tts(x,y, test_size=0.2, random_state=28)


# A glance at the shapes of train and test sets

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Training the model

LinReg=lreg()
LinReg.fit(x_train, y_train)
y_pred=LinReg.predict(x_test)


# Analysing the performance of the model
 
print('Test set (RMSE):', mean_squared_error(y_test, y_pred, squared=False))

print('NRMSE:', (mean_squared_error(y_test, y_pred, squared=False)/(y.max()-y.min())))

print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))


# Analysing the statistical significance

X_train_=x_train
y_train_=y_train
X_train_=sm.add_constant(X_train_)
Reg=sm.OLS(y_train_, X_train_).fit()
Reg.summary()

        """
        From these metrics it is noticed that there is not a linear relationship
        statistically significant between the Net Income and the factors considered
        in this project (namely Most Expressive Age Range Concentration,
        Training Hours and Education Level).
        """
______________________________________________end___________________________________________________