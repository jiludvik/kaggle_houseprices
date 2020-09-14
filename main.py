# KAGGLE HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES

#%% Load Libraries and Set Global Options - MANDATORY

#Essentials
import numpy as np
import pandas as pd
from datetime import timedelta
from time import perf_counter
import random

#Plotting
import seaborn as sns
import matplotlib.pyplot as plt

#Statistics
from scipy.stats import shapiro, normaltest,anderson
from pingouin import multivariate_normality

#Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV, RFE
from sklearn.pipeline import Pipeline

#Model Selection
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold

#Models
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

# Display all columns
pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
#warnings.filterwarnings(action="ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#%% Load data - MANDATORY
train = pd.read_csv('input/train.csv')
train.name ='train_data'
test = pd.read_csv('input/test.csv')
test.name='test_data'
all_datasets=[train,test]

train_labels = train['SalePrice'].reset_index(drop=True)
combined = pd.concat([train.drop(['SalePrice'], axis=1),test]).reset_index(drop=True)

print('Training Data Shape:',train.shape)
print('Test Data Shape:', test.shape)
print('Combined Data Shape:', combined.shape)
#%% Check test and train data set have the same columns and data types - MANDATORY

#Function to create a data frame with columns with mismatching names or data types
def schema_issues(df1=None,df2=None):
    df1_types=pd.DataFrame(df1.dtypes.astype(str), columns=['DF1'])
    df2_types=pd.DataFrame(df2.dtypes.astype(str), columns=['DF2'])
    result = pd.concat([df1_types, df2_types], axis=1)
    result['colname_issue'] = result.isnull().any(axis=1)
    result['dtype_issue'] = (result.DF1!=result.DF2)
    result=result.query('colname_issue or dtype_issue')
    return(result)

#report on differences in column names and dtypes between test & train data sets
print('Test and training data set columns with mismatched names or data types')
print(schema_issues(train,test))

#%% Summarise data - OPTIONAL

# Function to print unique values in each column of a data frame
def df_unique(df):
    for col in df:
        print(col, ': ', df[col].nunique(), ':', df[col].unique())

# Analyse numerical variables
train_describe=train.describe().transpose()
print('Max Values\n', train_describe.sort_values(by='min').head(10))
# Min: All numerical variables have positive values
print('Min Values\n', train_describe.sort_values(by='max', ascending=False).head(10))
# Max: Top columns
# LotArea - maximum 20x mean value, large std
# Miscval -  - weird distribution with zero in min/25%/50% and largish max
# Variables describing space: TotalBsmtSF, BsmtFinSF1, GrLivArea,1stFlrSF, BsmtUnfSF, 2ndFlrSF
print ('Standard Deviation\n', train_describe.sort_values(by='std', ascending=False).head(10))
# Std: similar to max

#%% IMPUTE MISSING VALUES, ENCODE LABELS & TRANSFORM DATA TYPES

# Define function to create a data frame with column data types, and missing value stats
def descriptive_df(df):
    ddf = pd.DataFrame({'column_name': df.columns,
                                 'data_type' : df.dtypes.astype(str),
                                 'count_missing': df.isnull().sum(),
                                 'percent_missing': df.isnull().sum() * 100 / len(df)})
    ddf.sort_values(['percent_missing','column_name'], inplace=True, ascending=False)
    return(ddf)

print('Before Transformation')
print(descriptive_df(combined))
print()

#MSZoning
combined['MSZoning'].isnull().sum()
combined['MSZoning'] = combined.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
#sns.boxplot(y=combined['MSSubClass'].astype(int), x=combined['MSZoning'])
#sns.boxplot(y=combined['MSSubClass'].astype(int), x=combined['Neighborhood'])
#plt.show()
combined[combined['MSZoning'].isnull()]

# MSSubClass, MoSold and YrSold: Convert to string
combined[['MSSubClass', 'MoSold', 'YrSold']].dtypes
combined[['MSSubClass', 'MoSold', 'YrSold']]=combined[['MSSubClass', 'MoSold', 'YrSold']].astype(str)
combined[['MSSubClass', 'MoSold', 'YrSold']].dtypes

#CentralAir
combined['CentralAir'].unique()
combined['CentralAir']=combined['CentralAir'].map({'N':0, 'Y':1})
combined['CentralAir']=combined['CentralAir'].astype(int)
combined['CentralAir'].unique()

# PoolQC: Fill NAs with 'None', map to ordinal values and convert to integer
combined['PoolQC'].isnull().sum()
combined['PoolQC'].unique()
combined['PoolQC']=combined['PoolQC'].fillna('None')
combined['PoolQC']=combined['PoolQC'].map({'None':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
combined['PoolQC']=combined['PoolQC'].astype(int)
combined['PoolQC'].isnull().sum()
combined['PoolQC'].unique()

#MiscFeature, Alley, Fence = fill NAs with 'None'
combined[['MiscFeature', 'Alley', 'Fence']].isnull().sum()
#df_unique(combined[['MiscFeature', 'Alley', 'Fence']])
combined[['MiscFeature', 'Alley', 'Fence']]=combined[['MiscFeature', 'Alley', 'Fence']].fillna('None')
combined[['MiscFeature', 'Alley', 'Fence']].isnull().sum()
#df_unique(combined[['MiscFeature', 'Alley', 'Fence']])

#FireplaceQu: Replace NAs with 'None', map to ordinal values and comvert to integer
combined['FireplaceQu'].isnull().sum()
combined['FireplaceQu'].unique()
combined['FireplaceQu']=combined['FireplaceQu'].fillna('None')
combined['FireplaceQu']=combined['FireplaceQu'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['FireplaceQu']=combined['FireplaceQu'].astype(int)
combined['FireplaceQu'].dtypes
combined['FireplaceQu'].isnull().sum()
combined['FireplaceQu'].unique()

#LotFrontage: Replace NAs with median value for the neighborhood
combined['LotFrontage'].isnull().sum()
combined['LotFrontage'] = combined.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
combined['LotFrontage'].isnull().sum()

#LotShape: Map to ordinal values and convert to integer
combined['LotShape'].isnull().sum()
combined['LotShape'].unique()
combined['LotShape']=combined['LotShape'].map({'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3})
combined['LotShape']=combined['LotShape'].astype(int)
combined['LotShape'].isnull().sum()
combined['LotShape'].unique()

#HeatingQC
combined['HeatingQC'].unique()
combined['HeatingQC']=combined['HeatingQC'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['HeatingQC']=combined['HeatingQC'].astype(int)

# STAGE 2 (GARAGE)

#print('Missing Values In Garage Variables')
#print(combined[['GarageYrBlt', 'GarageArea', 'GarageCars', 'GarageQual','GarageCond','GarageType','GarageFinish']].isnull().sum())
#print('\nUnique Values In Garage Categorical Variables')
#print(df_unique(combined[['GarageQual','GarageCond','GarageType','GarageFinish']]))

#GarageArea
combined['GarageArea'].isnull().sum()
combined['GarageArea']=combined['GarageArea'].fillna(combined['GarageArea'].mode().iloc[0])
combined['GarageArea'].isnull().sum()

#GarageCars
combined['GarageCars'].isnull().sum()
combined['GarageCars']=combined['GarageCars'].fillna(combined['GarageCars'].mode().iloc[0])
combined['GarageCars']=combined['GarageCars'].astype(int)
combined['GarageCars'].isnull().sum()

#Rows with GarageYrBlt missing
# Find rows that have GarageYrBlt==NA and GarageArea>0
index=combined[(combined['GarageYrBlt'].isnull()) & (combined['GarageArea']>0)].index
combined.loc[index,['GarageArea','GarageYrBlt','GarageQual','GarageCond']]
#Fix row: GarageYrBlt=YearBuilt, GarageQual= mode, GarageCond=mode
combined.loc[index,'GarageYrBlt'] = combined.loc[index,'YearBuilt']
combined.loc[index,'GarageQual']=combined['GarageQual'].mode().iloc[0]
combined.loc[index,'GarageCond']=combined['GarageCond'].mode().iloc[0]
combined.loc[index,['GarageArea','GarageYrBlt','GarageQual','GarageCond']]

#Properties with no garage
# Find remaining rows with missing GarageYrBlt, GarageType, GarageFinish
combined[['GarageYrBlt', 'GarageType', 'GarageFinish']].isnull().sum()

#Create a flag indicating whether a property has a garage
#combined['No_Garage']=combined['GarageYrBlt'].isnull()

# GarageYrBlt: Replace remaining NAs with 0 and convert to int
combined['GarageYrBlt'].isnull().sum()
combined['GarageYrBlt']=combined['GarageYrBlt'].fillna(0)
combined['GarageYrBlt']=combined['GarageYrBlt'].astype(int)
combined['GarageYrBlt'].isnull().sum()

# GarageType: Replace NAs with 'None'
combined['GarageType'].isnull().sum()
combined['GarageType'].unique()
combined['GarageType']=combined['GarageType'].fillna('None')
combined['GarageType'].isnull().sum()
combined['GarageType'].unique()

# GarageFinish: Replace NAs with 'None', cardinal to ordinal values and data type to integer
combined['GarageFinish'].isnull().sum()
combined['GarageFinish'].unique()
combined['GarageFinish']=combined['GarageFinish'].fillna('None')
combined['GarageFinish']=combined['GarageFinish'].map({'None':0, 'Unf':1, 'RFn':2, 'Fin':3})
combined['GarageFinish']=combined['GarageFinish'].astype(int)
combined['GarageFinish'].isnull().sum()
combined['GarageFinish'].unique()

# GarageQual: Replace NAs with 'None', cardinal to ordinal values and data type to integer
combined['GarageQual'].isnull().sum()
combined['GarageQual'].unique()
combined['GarageQual']=combined['GarageQual'].fillna('None')
combined['GarageQual']=combined['GarageQual'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['GarageQual']=combined['GarageQual'].astype(int)
combined['GarageQual'].unique()
combined['GarageQual'].isnull().sum()

# GarageCond: Replace NAs with 'None', cardinal to ordinal values and data type to integer
combined['GarageCond'].isnull().sum()
combined['GarageCond'].unique()
combined['GarageCond']=combined['GarageCond'].fillna('None')
combined['GarageCond']=combined['GarageCond'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['GarageCond']=combined['GarageCond'].astype(int)
combined['GarageCond'].isnull().sum()
combined['GarageCond'].unique()

#print('Missing Values In Garage Variables')
#print(combined[['GarageYrBlt','GarageQual','GarageCond','GarageType','GarageFinish']].isnull().sum())
#print('\nUnique Values In Garage Categoric Variables')
#print(df_unique(combined[['GarageQual','GarageCond','GarageType','GarageFinish']]))

#STAGE 3 - BASEMENT

#print('Missing Values In Basement Variables')
#print(combined[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2']].isnull().sum())
#print('\nUnique Values In Garage Categoric Variables')
#print(df_unique(combined[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2']]))

# TotalBsmtSF,'BsmtFinSF1,BsmtFinSF2,BsmtUnfSF
combined[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']].isnull().sum()
# Find out which rows have missing Basement square footage values
index=combined[combined[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']].isnull().sum(axis=1)>0].index
# Fix Square footage values: replace with 0 (ie there is no Basement)
combined.loc[index,['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']]=[0,0,0,0]
combined[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']].isnull().sum()

# BsmtQual
combined['BsmtQual'].isnull().sum()
combined['BsmtQual'].unique()
# Fill in BsmtQual for rows with TotalBSMTSF>0 with most frequent value
index=combined[(combined['BsmtQual'].isnull()) & (combined['TotalBsmtSF']>0)].index
combined.loc[index,['BsmtQual']]=combined['BsmtQual'].mode().iloc[0]
# Fill the rest (i.e. houses that do not have a basement) with None
combined['BsmtQual']=combined['BsmtQual'].fillna('None')
combined['BsmtQual']=combined['BsmtQual'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['BsmtQual']=combined['BsmtQual'].astype(int)
combined['BsmtQual'].isnull().sum()
combined['BsmtQual'].unique()

# BsmtCond
combined['BsmtCond'].isnull().sum()
combined['BsmtCond'].unique()
#Values with TotalBsmtSF>0 will be populated with most frequent value
index=combined[(combined['BsmtCond'].isnull()) & (combined['TotalBsmtSF']>0)].index
combined.loc[index,['BsmtCond']]=[combined['BsmtCond'].mode().iloc[0]] * len(index)
combined['BsmtCond'].isnull().sum()
#Remaining missing values (corresponding to properties with no basement) will get 'None'
combined['BsmtCond']=combined['BsmtCond'].fillna('None')
combined['BsmtCond']=combined['BsmtCond'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['BsmtCond']=combined['BsmtCond'].astype(int)
combined['BsmtCond'].isnull().sum()
combined['BsmtCond'].unique()

# BsmtExposure
combined['BsmtExposure'].isnull().sum()
combined['BsmtExposure'].unique()
# Rows with missing values corresponding to properties with basement will be populated by most frequent value
index=combined[(combined['BsmtExposure'].isnull()) & (combined['TotalBsmtSF']>0)].index
combined.loc[index,['BsmtExposure']]=combined['BsmtExposure'].mode().iloc[0]
combined['BsmtExposure'].isnull().sum()
# Remaining with missing values for the remaining properties will be populated with 'None'
combined['BsmtExposure']=combined['BsmtExposure'].fillna('None')
combined['BsmtExposure']=combined['BsmtExposure'].map({'None':0, 'No':0, 'Mn':1, 'Av':2, 'Gd':3})
combined['BsmtExposure']=combined['BsmtExposure'].astype(int)
combined['BsmtExposure'].isnull().sum()
combined['BsmtExposure'].unique()

#BsmtFinType1
combined['BsmtFinType1'].isnull().sum()
combined['BsmtFinType1'].unique()
# Missing values corresponding to properties with BsmtFinSF1>0 or BsmtUnfSF>0
combined[(combined['BsmtFinType1'].isnull()) & ((combined['BsmtFinSF1']>0)| (combined['BsmtUnfSF']>0))].index
# no values with SF>0 -> no specific fixing needed
# Remaining missing values for the remaining properties will be populated with 'None'
combined['BsmtFinType1']=combined['BsmtFinType1'].fillna('None')
combined['BsmtFinType1']=combined['BsmtFinType1'].map({'None':0, 'Unf':1, 'LwQ':1, 'Rec':2, 'BLQ':3, 'ALQ':4, 'GLQ':5})
combined['BsmtFinType1']=combined['BsmtFinType1'].astype(int)
combined['BsmtFinType1'].isnull().sum()
combined['BsmtFinType1'].unique()

#BsmtFinType2
combined['BsmtFinType2'].isnull().sum()
combined['BsmtFinType2'].unique()
# Missing values corresponding to properties with BsmtFinSF1>0 or BsmtUnfSF>0
index=combined[(combined['BsmtFinType2'].isnull()) & ((combined['BsmtFinSF2']>0)| (combined['BsmtUnfSF']>0))].index
combined.loc[index,['BsmtFinType2']]=combined['BsmtFinType2'].mode().iloc[0]
# Remaining missing values for the remaining properties will be populated with 'None'
combined['BsmtFinType2']=combined['BsmtFinType2'].fillna('None')
combined['BsmtFinType2']=combined['BsmtFinType2'].map({'None':0, 'Unf':1, 'LwQ':1, 'Rec':2, 'BLQ':3, 'ALQ':4, 'GLQ':5})
combined['BsmtFinType2']=combined['BsmtFinType2'].astype(int)
combined['BsmtFinType2'].isnull().sum()
combined['BsmtFinType2'].unique()

#BsmtFullBath, BsmtHalfBath
combined[['BsmtFullBath','BsmtHalfBath']].isnull().sum()
#df_unique(combined[['BsmtFullBath','BsmtHalfBath']])
combined['BsmtFullBath']=combined['BsmtFullBath'].fillna(combined['BsmtFullBath'].mode().iloc[0])
combined['BsmtHalfBath']=combined['BsmtHalfBath'].fillna(combined['BsmtHalfBath'].mode().iloc[0])
combined['BsmtHalfBath']=combined['BsmtHalfBath'].astype(int)
combined['BsmtFullBath']=combined['BsmtFullBath'].astype(int)
combined[['BsmtFullBath','BsmtHalfBath']].isnull().sum()
#df_unique(combined[['BsmtFullBath','BsmtHalfBath']])

#print('Missing Values In Basement Variables')
#print(combined[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2']].isnull().sum())
#print('\nUnique Values In Garage Categoric Variables')
#print(df_unique(combined[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2']]))

# STAGE 4 - REMAINDER OF VARIABLES
# MasVnrArea & MasVnrType
combined[['MasVnrType','MasVnrArea']].isnull().sum()
combined['MasVnrType'].unique()
#replace MasVnrType for property that have MasVnrAre>0 or not null by mode()
index=combined[(combined['MasVnrType'].isnull()) & ((combined['MasVnrArea']>0) | (combined['MasVnrArea'].notnull()))].index
combined.loc[index,['MasVnrType']]=combined['MasVnrType'].mode().iloc[0]
# For the rest of the properties, we will assume MasVnrArea=0 and MasVnrType=None
combined['MasVnrArea']=combined['MasVnrArea'].fillna(0)
combined['MasVnrType']=combined['MasVnrType'].fillna('None')
#sns.boxplot(x='MasVnrType',y='SalePrice',data=train), plt.show()
combined['MasVnrType']=combined['MasVnrType'].map({'None':0, 'BrkCmn':0, 'CBlock':1, 'BrkFace':2, 'Stone':3})
combined['MasVnrType']=combined['MasVnrType'].astype(int)
combined[['MasVnrType','MasVnrArea']].isnull().sum()
combined['MasVnrType'].unique()

#KitchenQual
combined['KitchenQual'].isnull().sum()
combined['KitchenQual'].unique()
combined['KitchenQual']=combined['KitchenQual'].fillna(combined['KitchenQual'].mode().iloc[0])
combined['KitchenQual']=combined['KitchenQual'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['KitchenQual']=combined['MasVnrType'].astype(int)
combined['KitchenQual'].isnull().sum()
combined['KitchenQual'].unique()

#Utilities
combined=combined.drop('Utilities',axis=1)

#Functional
combined['Functional'].isnull().sum()
combined['Functional'].unique()
combined['Functional']=combined['Functional'].fillna(combined['Functional'].mode().iloc[0])
combined['Functional']=combined['Functional'].map({'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7})
combined['Functional']=combined['Functional'].astype(int)
combined['Functional'].isnull().sum()
combined['Functional'].unique()

#Exterior1st, Exterior2nd, ExteriorQual
combined[['Exterior1st','Exterior2nd','ExterQual', 'ExterCond']].isnull().sum()
combined['Exterior1st']=combined['Exterior1st'].fillna(combined['Exterior1st'].mode().iloc[0])
combined['Exterior2nd']=combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode().iloc[0])
combined['ExterQual']=combined['ExterQual'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['ExterCond']=combined['ExterCond'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
combined['ExterQual']=combined['ExterQual'].astype(int)
combined['ExterCond']=combined['ExterCond'].astype(int)
#df_unique(combined[['Exterior1st','Exterior2nd','ExterQual', 'ExterCond']])

#Electrical
combined['Electrical'].isnull().sum()
combined['Electrical'].unique()
combined['Electrical']=combined['Electrical'].fillna(combined['Electrical'].mode().iloc[0])
combined['Electrical'].isnull().sum()

#SaleType
combined[['SaleType']].isnull().sum()
combined['SaleType']=combined['SaleType'].fillna(combined['SaleType'].mode().iloc[0])
combined[['SaleType']].isnull().sum()

#Summarise missing values and data types
print('After Transformation')
print(descriptive_df(combined))
print()

#%% Recreate training and test data sets after imputation and label encoding
# training dataset with imputed values and encoded labels
train2_x = combined.iloc[:len(train_labels), :]
train2_y=train['SalePrice']
train2=pd.concat([train2_y, train2_x.reindex(train2_y.index)], axis=1)
print('Training data shape before / after transformation:', train.shape,'/', train2.shape)

# test data set
test2 = combined.iloc[len(train_labels):, :]
print('Test data shape before / after transformation:', test.shape,'/', test2.shape)

#%% Identify top features with Correlation Matrix- MANDATORY

# Correlation Matrix for Numerical Columns Only
def heatmap(data, title=None, annot=True, annot_fontsize=12):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        data,
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=annot,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': annot_fontsize}
    )
    plt.title(title, y=1.05, size=15)

# Helper function to fill diagonal of correlation matrix with 0
def df_fill_diagonal(df, val, wrap=False):
    arr=np.array(df)
    np.fill_diagonal(a=arr,val=val,wrap=wrap)
    return(pd.DataFrame(arr, index=df.index, columns=df.columns))

# Function to filter correlation matrix to include only values beyond certain thershold
def filter_corrmatrix(corr, corr_threshold):
    corr=df_fill_diagonal(corr,0)
    filtered = corr[(corr.abs() >= corr_threshold).any(axis=1)]
    filtered = filtered[filtered.columns[(filtered.abs() >= corr_threshold).any()]]
    filtered = df_fill_diagonal(filtered, 1)
    return(filtered)

num_cols = train2.select_dtypes(exclude='object').columns
corr_matrix = filter_corrmatrix (train2[num_cols].corr(),0.5)
heatmap(corr_matrix, title='Correlation - Numerical Features', annot_fontsize=8)
plt.show()

#Top numerical variables correlated with SalePrice (corr>0.6):
# OverallQual, GrLivArea, ExterQual, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF

#Mutual correlations - NEED REVISITING IF USED TO CLEAN UP DATA BEFORE MODELLING
# Dif types of square Footage
# Square Footage & No of Rooms
# Garage variables

#%% Identify top features with RFE- MANDATORY

#Generate a temporary data frame with remainder of categorical variables converted to dummy vars
# this is just temporary, as we will need to make further conversions before generating final version of the encoded dataframe
train3_x= pd.get_dummies(train2.drop(['SalePrice','Id'], axis=1), drop_first=True)
train3_y=train2['SalePrice']
train3=pd.concat([train3_y, train3_x.reindex(train3_y.index)], axis=1)

# Top features By RFE
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
top_features_no=10
selector_rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=top_features_no)
selector_rfe.fit(train3_x,train3_y)
best_features_rfe=train3_x.columns[selector_rfe.support_]
print ('Top', top_features_no,'Predictors by RFE with DecisionTreeRegressor')
print(best_features_rfe)

# LotFrontage, LotArea, OverallQual, YearBuilt, BsmtFinSF1,
# TotalBsmtSF,1stFlrSF, 2ndFlrSF, GrLivArea, GarageCars

# Optimum feature selection using RFECV
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
time_start = perf_counter()
scoring_method='neg_mean_squared_error'
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
model = DecisionTreeRegressor()
selector_rfecv = RFECV(estimator=model, scoring=scoring_method, n_jobs=-1, cv=cv_method)
selector_rfecv.fit(train3_x,train3_y)
time_stop = perf_counter()
best_features_rfecv=train3_x.columns[selector_rfecv.support_]
print('Best Features By RFECV with DecisionTreeClassifier')
print (len(best_features_rfecv),'out of', len(train3_x.columns),'columns')
print (best_features_rfecv)
print ('Elapsed time RFECV:', timedelta(seconds=round(time_stop-time_start,0)))

#%% Exploratory Data Analysis

#Topics / Functions for EDA
#   Is the variable normally distributed? :
#       Histogram / skew + kurtosis measures,
#       Shapiro-wilk or pearson chi squared
#   Relationship between the predictor and the response:
#       Scatter plot for continuous, bar plot for factors
#   Can/should the variable be grouped with others?:
#       Scatter plot / bar plot for all variables in the group


def mydistplot(x, data=None, x_label=None, y_label=None, title=None, show=True):
    plot = sns.distplot(data[x]);
    plot.set(ylabel=y_label)
    plot.set(xlabel=x_label)
    plot.set(title=title)
    sns.despine(trim=True, left=True)
    if show:
        plt.show()
    else:
        return(plot)

def myboxplot(x=None, y=None, data=None, title=None, h_line=None, show=True):
    plot = sns.boxplot(x=data[x], y=data[y])
    if h_line !=None:
        plot.axhline(h_line)
    plot.set(title=title)
    if show:
        plt.show()
    else:
        return(plot)

def mylmplot(x=None, y=None, data=None, title=None, show=True):
    plot3 = sns.lmplot(x=x, y=y,
                       lowess=True,
                       scatter_kws={'alpha': 0.05},
                       line_kws={'color': 'darkblue'},
                       data=data)
    plot3.set(title=title)
    if show:
        plt.show()
    else:
        return(plot)

#SALEPRICE
mydistplot('SalePrice', train, y_label="Frequency", title="SalePrice Distribution", show=True)

#SALEPRICE BY NEIGBOURGHOOD
plot=sns.stripplot(x=train['Neighborhood'], y=train['SalePrice'], hue=train['OverallQual'].tolist(), alpha=.25 )
plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
plot.axhline(train['SalePrice'].mean())
plot.set(title="SalePrice Per Neighborhood & Overall Quality")
plt.show()

#SALEPRICE BY OVERALLQUAL
myboxplot (x='OverallQual',y='SalePrice', data=train, title='SalePrice by OverallQual', h_line=train['SalePrice'].mean())

#SALEPRICE BY YEARBUILT
mylmplot(x='YearBuilt', y='SalePrice', data=train, title="SalePrice By YearBuilt")

#SALEPRICE BY LOTFRONTAGE/LOTAREA
mylmplot(x='SalePrice', y='LotFrontage', data=train, title="LotFrontage by SalePrice")
mylmplot(x='SalePrice', y='LotArea', data=train, title="LotArea by SalePrice")

# SalePrice by Square Footage
mylmplot(y='SalePrice', x='GrLivArea', data=train, title="SalePrice by GrLivArea")
mylmplot(y='SalePrice', x='TotalBsmtSF', data=train, title="SalePrice by TotalBsmtSF")
mylmplot(y='SalePrice', x='BsmtFinSF1', data=train, title="SalePrice by BsmtFinSF1")
mylmplot(y='SalePrice', x='1stFlrSF', data=train, title="SalePrice by 1stFlrSF")
mylmplot(y='SalePrice', x='2ndFlrSF', data=train, title="SalePrice by 2ndFlrSF")

# Feature engineering ideas
# - Combine SF and have flags for having / not having basement or 2nd Floor or Garage
# - Replace neighborougood with average square foot sale price / lotarea sale price??

#SALEPRICE BY GarageCars & GarageArea
myboxplot (x='GarageCars',y='SalePrice', data=train, title='SalePrice by GarageCars', h_line=train['SalePrice'].mean())
mylmplot(x='GarageArea', y='SalePrice', data=train, title="SalePrice by GarageArea")

# SalePrice goes down after 4th car - most people have max 4 cars?
# or no-one is recording car space beyond 5 cars
# Question: Zero values in 2nd FlrSF, BsmtFinSF1, GarageArea are skewing distribution


#%% Normality test of all variables

print('Multivariate Normality Test:')
stat, p, normal = multivariate_normality(train.select_dtypes(exclude='object').to_numpy())
print('Statistics=%.3f, p=%.3f' % (stat, p))
print('Does data set have normal distribution:', normal)


#%% TOMORROW
# 3) Feature engineering
# Go through both notebooks and pick up useful ideas
# Try to run column clustering on the data set?

# 2) Go through categorical values with low number of observations


# 4)  Outlier identification: method tbc
#https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
