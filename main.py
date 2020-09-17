# KAGGLE HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES

#%% LOAD LIBRARIES - MANDATORY

#Data Wrangling
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, RobustScaler
from datetime import timedelta
from time import perf_counter

#Exploratory Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

#Modelling
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.metrics import mean_squared_error

# Display all columns
pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
#warnings.filterwarnings(action="ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#%% LOAD DATA - MANDATORY

# load data
train = pd.read_csv('input/train.csv')
train.name ='train_data'
test = pd.read_csv('input/test.csv')
test.name='test_data'

train_labels = train['SalePrice'].reset_index(drop=True)
combined = pd.concat([train.drop(['SalePrice'], axis=1),test]).reset_index(drop=True)

print('Training Data Shape:',train.shape)
print('Test Data Shape:', test.shape)
print('Combined Data Shape:', combined.shape)

#Function to create a data frame with columns with mismatching names or data types
def schema_issues(df1=None,df2=None):
    df1_types=pd.DataFrame(df1.dtypes.astype(str), columns=['DF1'])
    df2_types=pd.DataFrame(df2.dtypes.astype(str), columns=['DF2'])
    result = pd.concat([df1_types, df2_types], axis=1)
    result['colname_issue'] = result.isnull().any(axis=1)
    result['dtype_issue'] = (result.DF1!=result.DF2)
    result=result.query('colname_issue or dtype_issue')
    return(result)

#Report on differences in column names and dtypes between test & train data sets
print('Test and training data set columns with mismatched names or data types')
print(schema_issues(train,test))

#%% SUMMARISE DATA - OPTIONAL

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

#%% IMPUTE MISSING VALUES & ENCODE ORDINAL VALUES  - MANDATORY

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

# GARAGE VARIABLES

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
missinggarage_row=combined[(combined['GarageYrBlt'].isnull()) & (combined['GarageArea']>0)].index
combined.loc[missinggarage_row,['GarageArea','GarageYrBlt','GarageQual','GarageCond']]
#Fix row: GarageYrBlt=YearBuilt, GarageQual= mode, GarageCond=mode
combined.loc[missinggarage_row,'GarageYrBlt'] = combined.loc[missinggarage_row,'YearBuilt']
combined.loc[missinggarage_row,'GarageQual']=combined['GarageQual'].mode().iloc[0]
combined.loc[missinggarage_row,'GarageCond']=combined['GarageCond'].mode().iloc[0]
combined.loc[missinggarage_row,['GarageArea','GarageYrBlt','GarageQual','GarageCond']]

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

#BASEMENT VARIABLES

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

# REMAINDER OF VARIABLES
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
combined['KitchenQual']=combined['KitchenQual'].astype(int)
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

#Street
combined['Street'].unique()
combined['Street']=combined['Street'].map({'Grvl':0, 'Pave':1})
combined['Street']=combined['Street'].astype(int)

#PavedDrive
combined['PavedDrive'].unique()
combined['PavedDrive']=combined['PavedDrive'].map({'N':0, 'P':1, 'Y':2})
combined['PavedDrive']=combined['PavedDrive'].astype(int)
combined['PavedDrive'].unique()

#Id
combined=combined.drop('Id', axis=1)

#Summarise missing values and data types
print('After Transformation')
print(descriptive_df(combined))
print()

#%% FEATURE ENGINEERING - MANDATORY
#https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda and

combined['TotalBathrooms']=combined['FullBath']+combined['BsmtFullBath']+0.5*combined['HalfBath']+0.5*combined['BsmtHalfBath']
yrsold_issue_index=combined[combined['YrSold'].astype(int)<combined['YearBuilt']].index
combined.loc[yrsold_issue_index,'YrSold']=combined.loc[yrsold_issue_index,'YearBuilt']
combined['Age']=combined['YrSold'].astype(int)-combined['YearBuilt']
neighborhood_map={'MeadowV':0, 'IDOTRR':0, 'BrDale':0,
              'StoneBr':2, 'NridgHt':2, 'NoRidge':2,
              'CollgCr':1, 'Veenker':1, 'Crawfor':1,  'Mitchel':1, 'Somerst':1,'NWAmes':1,
              'OldTown':1, 'BrkSide':1, 'Sawyer':1,  'NAmes':1,'SawyerW':1,  'Edwards':1,
              'Timber':1, 'Gilbert':1,'ClearCr':1, 'NPkVill':1, 'Blmngtn':1,  'SWISU':1,'Blueste':1}

combined['NeighborAffl']=combined['Neighborhood'].map(neighborhood_map)
combined['TotalSqFeet']=combined['GrLivArea']+combined['TotalBsmtSF']
combined['TotalPorchSF']=combined['OpenPorchSF'] + combined['EnclosedPorch'] + combined['3SsnPorch'] + combined['ScreenPorch']

#generate binary variables
combined['IsRemodelled']=(combined['YearBuilt']!=combined['YearRemodAdd'])
combined['NewBuild']=(combined['YrSold']==combined['YearBuilt'])
combined['HasPool'] = combined['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
#sns.boxplot(x=train2['HasPool'], y=train2['SalePrice']), plt.show()
combined['HasGarage'] = combined['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#sns.boxplot(x=train2['HasGarage'], y=train2['SalePrice']), plt.show()
combined['HasBsmt'] = combined['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#sns.boxplot(x=train2['HasBsmt'], y=train2['SalePrice']), plt.show()
combined['HasFireplace'] = combined['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#sns.boxplot(x=train2['HasFireplace'], y=train2['SalePrice']), plt.show()
combined['Has2ndFloor'] = combined['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#Create an index with all binary variables required for further data transformations
binary_numcols=['IsRemodelled', 'NewBuild','HasPool','HasGarage',
                'HasBsmt','HasFireplace','Has2ndFloor','CentralAir','Street']

#%% RECREATE TRAINING DATASET FOR THE PURPOSE OF EDA - MANDATORY

# training dataset with imputed values and encoded labels
train2_x = combined.iloc[:len(train_labels), :]
train2_y=train['SalePrice']

train2=pd.concat([train2_y, train2_x.reindex(train2_y.index)], axis=1)
print('Training data shape before / after imputation & feat. eng.:', train.shape,'/', train2.shape)
train2_x.shape

# test data set with imputed values and encoded labels
test2 = combined.iloc[len(train_labels):, :]
print('Test data shape before / after imputation & feat.eng.:', test.shape,'/', test2.shape)

#%% EDA: IDENTIFY TOP NUMERICAL FEATURES WITH CORR.MATRIX & RFE - MANDATORY

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

#Top variables correlated with SalePrice (corr>0.6) Before Feature Engineering:
# OverallQual, GrLivArea, ExterQual, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF
#Top variables correlated with SalePrice (corr>0.6) After Feature Engineering:
#OverallQual, TotalSqFeet,GrLivArea,ExterQual,GarageCars, TotalBathrooms, GarageArea,1stFloorSF, TotalBsmtSF

# Top features By RFE
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
top_features_no=10
selector_rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=top_features_no)
train2num_x=train2_x.select_dtypes(exclude='object')
selector_rfe.fit(train2num_x,train2_y)
best_features_rfe=train2num_x.columns[selector_rfe.support_]
print ('Top', top_features_no,'Predictors by RFE with DecisionTreeRegressor')
print(best_features_rfe)

#LotArea', 'OverallQual', 'YearRemodAdd', 'BsmtFinSF1', 'TotalBsmtSF',
#       '2ndFlrSF', 'GrLivArea', 'GarageArea', 'Age', 'TotalSqFeet'

#%% EDA: RELATIONSHIP BETWEEN TOP VARIABLES - OPTIONAL

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

#%% REMOVE OUTLIERS - MANDATORY
# create new transformed combined and training data sets (with outliers removed)

outlier_rows=train2[(train2['OverallQual']<5) & (train2['SalePrice']>200000)].index.\
    union(train2[(train2['GrLivArea']>4500) & (train2['SalePrice']<300000)].index)
print('Outlier rows:', list(outlier_rows))

#remove outliers from the combined and train datasets
combined_trf=combined.copy().drop(outlier_rows).reset_index(drop=True)
train_trf=train2.copy().drop(outlier_rows).reset_index(drop=True)

#check the results
print('Combined imputed & encoded train+test features before/after outlier removal:',
      combined.shape, combined_trf.shape)
print('Imputed & label encoded training data before/after outlier removal:',
      train2.shape, train_trf.shape)


#%% SCALE AND ADDRESS SKEWNESS IN PREDICTORS+ RESPONSE- MANDATORY

#https://medium.com/@sjacks/feature-transformation-21282d1a3215
#https://www.kaggle.com/jiriludvik/how-i-made-top-0-3-on-a-kaggle-competition/edit?rvi=1
#https://www.datasklr.com/ols-least-squares-regression/transforming-variables

#Function to create log scale box plot for all variables in a df
def boxplot_logscale(df):
    sns.set_style("white")
    f, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale("log")
    ax = sns.boxplot(data=df , orient="h", palette="Set1")
    ax.xaxis.grid(False)
    ax.set(ylabel="Feature names")
    ax.set(xlabel="Numeric values")
    ax.set(title="Numeric Distribution of Features")
    sns.despine(trim=True, left=True)
    plt.show()

#Function to identify all variables in a df with skewness> threshold
def get_skewed_index(df, thresh=.5):
    skew_features = df.apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skew_features[skew_features > thresh]
    skew_index = high_skew.index
    print("There are {} numerical features with Skew > threshold:".format(high_skew.shape[0]))
    return(skew_index)

# Create an index of numerical columns w/o binary columns
num_cols=list(set(combined_trf.select_dtypes(exclude='object').columns)-set(binary_numcols))

#Describe features before transformations
combined_trf[num_cols].describe().transpose()[['min','max','mean','std']]
boxplot_logscale(combined_trf[num_cols])
# biggest max is five orders of magnitude bigger than the smallest max - needs scaling

#Scale & center data
scaler=RobustScaler()
combined_trf[num_cols]=scaler.fit_transform(combined_trf[num_cols])
combined_trf[num_cols].describe().transpose()[['min','max','mean','std']]
boxplot_logscale(combined_trf[num_cols])

# Fix skewed features
skewed_cols=get_skewed_index(combined_trf[num_cols])
print(skewed_cols)
transformer=PowerTransformer(method='yeo-johnson')
combined_trf[skewed_cols]=transformer.fit_transform(combined_trf[skewed_cols])
skewed_cols=get_skewed_index(combined_trf[num_cols])
print(skewed_cols)
combined_trf[skewed_cols].hist(), plt.show()
# 'PoolQC', 'PoolArea', '3SsnPorch', 'LowQualFinSF', 'MiscVal',
#       'BsmtHalfBath', 'ScreenPorch', 'BsmtFinSF2', 'EnclosedPorch',
#       'BsmtExposure', 'MasVnrArea', 'HalfBath'

#drop remaining skewed variables
combined_trf=combined_trf.drop(skewed_cols, axis=1)

#recalculate numerical columns
num_cols=list(set(combined_trf.select_dtypes(exclude='object').columns)-set(binary_numcols))

#report on columns with skewed features
get_skewed_index(combined_trf[num_cols])

#report on numerical columns
boxplot_logscale(combined_trf[num_cols])

# Relatively large number of cardinal variables encoded to ordinal scale are skewed.
# In the next version of the script, it may be better to keep them as cardinals and simply generate dummy vars

# Transform response to log scale - this will need to be taken into consideration when generating predictions
mydistplot('SalePrice',train_trf)
train_trf["SalePrice"] = np.log1p(train_trf["SalePrice"])
mydistplot('SalePrice',train_trf)

#%% GENERATE DUMMY VARIABLES FROM REMAINING FACTORS - MANDATORY

# Generate dummy variables
combined_trf2=pd.get_dummies(combined_trf).reset_index(drop=True)

# Address duplicate column names
# check duplicate column names
cols=pd.Series(combined_trf2.columns)
dup_names=cols[cols.duplicated()].unique()
print('Duplicate column names:', dup_names)

#creat a list with transformed col names (duplicate names have a sequence number appended)
for dup in dup_names:
    cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
#rename duplicate column names with the new list of unique column names
combined_trf2.columns=cols

# replace duplicate columns by a new column created by row-wise addition of values from orig columns
for dup in dup_names:
    print('Treating column', dup)
    dup_mask=cols.str.startswith(dup)
    dup_pair=combined_trf2.columns[dup_mask]
    print('Sum\n', combined_trf2[dup_pair].sum())
    rowsum=combined_trf2[dup_pair].sum(axis=1)
    if (rowsum>1).sum()==0:
        combined_trf3=combined_trf2.drop(dup_pair,axis=1)
        combined_trf3[dup]=rowsum
    else:
        print ('Data issues in columns {}, proceed with manual de-duplication'.format(dup))

print('Remaining duplicate column names:',combined_trf3.columns[combined_trf3.columns.duplicated()])
combined_trf3.shape
combined_trf2.shape

print('Combined data set before/after creation of dummy variables:', combined_trf.shape, combined_trf3.shape)

#%% RECREATE TRAINING AND TEST DATA SETS AFTER TRANSFORMATION - MANDATORY

# Transformed training dataset
train3_x = combined_trf3.iloc[:len(train_trf), :]
train3_y=train_trf['SalePrice']
print('Training features before / after transformation:', train2_x.shape,'/', train3_x.shape)
print('Training response before / after transformation:', train2_y.shape,'/', train3_y.shape)

train3=pd.concat([train3_y, train3_x.reindex(train3_y.index)], axis=1)
print('Training features+response before / after imputation & feat. eng.:', train2.shape,'/', train3.shape)

# test data set with imputed values and encoded labels
test3 = combined_trf3.iloc[len(train_trf):, :]
print('Test features before / after transformation:', test2.shape,'/', test3.shape)

#%% MODELLING: DEFINE MODELS
#To save time, largely reused https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition

# Define cross validation and scoring methods
scoring_method='neg_root_mean_squared_error'
cv_method=KFold(n_splits=13, random_state=0, shuffle=True)

# Define function that returns positive RMSE
def get_cv_score(model, X=train3_x, y=train3_y):
    rmse = -cross_val_score(model, X, y, scoring=scoring_method, cv=cv_method)
    return (rmse)

lasso= Lasso(alpha=0.000506,
             max_iter=10000,
             random_state=0)

lightgbm = LGBMRegressor(objective='regression',
                       num_leaves=6,
                       learning_rate=0.01,
                       n_estimators=7000,
                       max_bin=200,
                       bagging_fraction=0.8,
                       bagging_freq=4,
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                         n_jobs=-1,
                       verbose=-1,
                       random_state=0)

xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       nthread=-1,
                       n_jobs=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=0)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = RidgeCV(alphas=ridge_alphas, cv=cv_method)

# Support Vector Regressor
svr = SVR(C= 20, epsilon= 0.008, gamma=0.0003)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=0)

rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=0,
                          n_jobs=-1)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(lasso, xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

def elapsed_time(time_start, time_stop):
    print ('Elapsed time:', timedelta(seconds=round(time_stop-time_start,0)))

#%% MODELLING : GET BASELINE CROSS-VALIDATION SCORES - MANDATORY

scores = {}

print('Baseline Cross-Validation Scores (RMSE)')

print('lasso')
t_start = perf_counter()
score = get_cv_score(lasso)
t_stop = perf_counter()
print("lasso: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['las'] = (score.mean(), score.std())
elapsed_time(t_start,t_stop)

print('\nlightgbm')
t_start = perf_counter()
score = get_cv_score(lightgbm)
t_stop = perf_counter()
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())
elapsed_time(t_start,t_stop)

print('\nxgboost')
t_start = perf_counter()
score = get_cv_score(xgboost)
t_stop = perf_counter()
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())
elapsed_time(t_start,t_stop)
#
print('\nsvr')
t_start = perf_counter()
score = get_cv_score(svr)
t_stop = perf_counter()
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
elapsed_time(t_start,t_stop)

#
print('\nridge')
t_start = perf_counter()
score = get_cv_score(ridge)
t_stop = perf_counter()
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())
elapsed_time(t_start,t_stop)

print('\ngradient boosting')
t_start = perf_counter()
score = get_cv_score(gbr)
t_stop = perf_counter()
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
elapsed_time(t_start,t_stop)

#%%
#%TRAIN MODELS
print('Training Models\n')
print('stack_gen')
t_start = perf_counter()
stack_gen_model = stack_gen.fit(np.array(train3_x), np.array(train3_y))
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('lasso')
t_start = perf_counter()
las_model_full_data = lasso.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('lightgbm')
t_start = perf_counter()
lgb_model_full_data = lightgbm.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('xgboost')
t_start = perf_counter()
xgb_model_full_data = xgboost.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('svr')
t_start = perf_counter()
svr_model_full_data = svr.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('ridge')
t_start = perf_counter()
ridge_model_full_data = ridge.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('randomforest')
t_start = perf_counter()
rf_model_full_data = rf.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

print('GradientBoosting')
t_start = perf_counter()
gbr_model_full_data = gbr.fit(train3_x, train3_y)
t_stop = perf_counter()
elapsed_time(t_start,t_stop)

#%%

# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.1 * ridge_model_full_data.predict(X)) + \
            (0.2 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.1 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.05 * rf_model_full_data.predict(X)) + \
            (0.35 * stack_gen_model.predict(np.array(X))))

stacked_score=mean_squared_error(y_true=train3_y, y_pred=stack_gen_model.predict(np.array(train3_x)), squared=False)
print('stacked:', stacked_score)
scores['stacked'] = (stacked_score, 0)

blended_score = mean_squared_error(y_true=train3_y, y_pred=blended_predictions(train3_x), squared=False)
scores['blended'] = (blended_score, 0)
print('blended:', blended_score)


#%%  Plot the predictions for each model

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', labelpad=12.5)
plt.xlabel('Model', labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.show()

#%% Prepare submission
# Read in sample_submission dataframe

submission = pd.read_csv("input/sample_submission.csv")
submission.shape

# Append predictions from blended models
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(test3)))
#submission.iloc[:,1] = np.floor(np.expm1(stack_gen_model.predict(np.array(test3))))
#submission.iloc[:,1] = np.floor(np.expm1(svr_model_full_data.predict(np.array(test3))))


# Fix outlier predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

# Scale predictions
submission['SalePrice'] *= 1.001619
submission.shape
#submission.to_csv("output/submission_regression_stack.csv", index=False)
submission.to_csv("output/submission_regression_blend.csv", index=False)
#submission.to_csv("output/submission_regression_svr.csv", index=False)

#Blended model:  758 on leaderboard - RMSLE 0.12239
