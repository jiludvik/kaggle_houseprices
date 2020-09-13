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

#Preprocessing
from scipy.stats import shapiro
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFECV, RFE
from sklearn.pipeline import Pipeline

#Model Selection
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold

#Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

#Statistics


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
combined = pd.concat([train.drop(['SalePrice'], axis=1),test]).reset_index(drop=True)
print('Training Data Shape:',train.shape)
print('Test Data Shape:', test.shape)
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

# Analyse character columns (Object type
combined_objectcols=combined.select_dtypes(include='object')
df_unique(combined_objectcols)


#%% Fix incorrectly inferred data types - MANDATORY

# Columns with incorrectly inferred data types
#target_colnames=['OverallCond', 'OverallQual', 'MSSubClass', 'MoSold', 'YrSold', 'GarageYrBlt','YearBuilt','YearRemodAdd']
#target_colnames=['OverallCond', 'OverallQual', 'MSSubClass']
target_colnames=['MSSubClass', 'MoSold']

target_dtype=str

# Fix incorrectly inferred data types in 'MoSold', 'YrSold', 'OverallCond', 'MSSubClass' columns
for dataset in all_datasets:
    dataset[target_colnames]=dataset[target_colnames].astype(target_dtype)


#%% Correlation Matrix for Numerical Columns Only - OPTIONAL
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

# Function to filter correlation matrix to include only values beyond certainthershol
def filter_corrmatrix(corr, corr_threshold):
    corr=df_fill_diagonal(corr,0)
    filtered = corr[(corr.abs() >= corr_threshold).any(axis=1)]
    filtered = filtered[filtered.columns[(filtered.abs() >= corr_threshold).any()]]
    filtered = df_fill_diagonal(filtered, 1)
    return(filtered)

num_cols = train.select_dtypes(exclude='object').columns
corr_matrix = filter_corrmatrix (train[num_cols].corr(),0.5)
heatmap(corr_matrix, title='Correlation - Numerical Features')
plt.show()

#Top numerical variables correlated with SalePrice are"
# OverallQual, GrLivArea, GarageCars+GarageArea, TotalBsmtSF+1stFlrSF

# Floor Space is mutually correlated, Garage variables are mutually correlated

#%% Analyse missing values - OPTIONAL

# Define function to create a data frame with column data types, and missing value stats
def descriptive_df(df):
    #df=test
    ddf = pd.DataFrame({'column_name': df.columns,
                                 'data_type' : df.dtypes.astype(str),
                                 'count_missing': df.isnull().sum(),
                                 'percent_missing': df.isnull().sum() * 100 / len(df)})
    ddf.sort_values('percent_missing', inplace=True, ascending=False)
    return(ddf)

# Create dataframes with descriptive stats
train_desc_df=descriptive_df(train)
train_desc_df.name ='Training data'
test_desc_df=descriptive_df(test)
test_desc_df.name ='Test data'
combined_desc_df=descriptive_df(combined)
combined_desc_df.name ='Training + test data'

# Look at columns with missing values in test & train data set
print('Analysis of missing values')
for df in [train_desc_df, test_desc_df]:
    print(df.name)
    print('----------')
    print(df.query('count_missing > 0'))
    print()

#%% Define Imputation Strategy - IGNORE
# THIS IS NOT A GOOD IDEA FOR A DATAFRAME WITH THE SHAPE of 1500x80. MAYBE FOR 10xLARGER DATASET

# Function to evaluate an (independent) feature and recommend imputation strategy
def imputation_strat_ind(feature=None, dataset=train):
    if dataset[feature].isnull().sum()==0:
        action='Do Nothing'
    elif feature in ['Alley','Fence','MiscFeature']:
        action='Replace With None'
    elif feature=='LotFrontage':
        action='Depends on LotConfig'
    elif feature=='MasVnrArea':
        action='Replace With 0'
    else:
        action='Impute'
    return(action)

# Function to evaluate a pair of dependent features and recommend imputation strategy
def imputation_strat_dep(ind_feature=None, dep_feature=None, baseline=0, dataset=train, verbose=False):
   case2_count=dataset[dataset[ind_feature]!=baseline][dep_feature].isnull().sum()
   case1_count=dataset[dataset[ind_feature]==baseline][dep_feature].isnull().sum()
   if case1_count!=0:
       action='Replace with None'
   elif case2_count!=0:
       action='Impute'
   else:
       action='Do Nothing'
   if verbose:
       print('Variable', dep_feature)
       print ('Count ({0}==0 & {1} == Null) = {2}'.format(ind_feature, dep_feature,case1_count))
       print('Count ({0}!=0 & {1} == Null) = {2}'.format(ind_feature, dep_feature, case2_count))
       print ('Cleansing Approach:',action)
   return(action)

# Identify cleansing value for variables whose Null value may be dependent on value of other variable
def desc_df_imputation(dataset, desc_df):
    desc_df['imp_strategy'] = np.nan
    for key in dep_features:
        for value in dep_features[key]:
            if value in dataset.columns:
                desc_df.loc[desc_df.column_name == value, 'imp_strategy'] = \
                    imputation_strat_dep(ind_feature=key, dep_feature=value, dataset=dataset)
    for value in ind_features:
        if value in dataset.columns:
            action=imputation_strat_ind(feature=value, dataset=dataset)
            desc_df.loc[desc_df.column_name == value, 'imp_strategy'] = action
    print()

ind_features = ['MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
       'MasVnrArea', 'MSZoning', 'Functional', 'BsmtHalfBath',
       'BsmtFullBath', 'Utilities', 'KitchenQual', 'BsmtFinSF1',
       'SaleType', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Exterior2nd',
       'Exterior1st', 'GarageArea', 'Electrical', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Fireplaces', 'Id', 'HalfBath', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
       'BedroomAbvGr', 'HeatingQC', 'FullBath', 'OverallCond', 'LotArea',
       'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'YearBuilt', 'GrLivArea', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
       'MSSubClass', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'SaleCondition','SalePrice']

# Define pairs of independent and dependent features to check
dep_features ={
    'PoolArea':['PoolQC'],
    'Fireplaces': ['FireplaceQu'],
    'GarageArea': ['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars'],
    'TotalBsmtSF': ['BsmtFinSF1','BsmtFinSF2','BsmtExposure','BsmtCond','BsmtQual','BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath'],
    'BsmtFinSF1': ['BsmtFinType1'],
    'BsmtFinSF2': ['BsmtFinType2'],
    'MasVnrArea': ['MasVnrType'],
    'LotArea': ['LotFrontage'],
    'KitchenAbvGr': ['KitchenQual']
}

desc_df_imputation(train, train_desc_df)
desc_df_imputation(test, test_desc_df)

#Generate indices of columns to action
for df in [train_desc_df,test_desc_df]:
    print(df.name)
    print('----------')
    print('Impute:',df.query('imp_strategy=="Impute"').index)
    print('Replace with "None":', df.query('imp_strategy=="Replace With None"').index)
    print('Replace with 0:', df.query('imp_strategy=="Replace With 0"').index)
    print('Depends on LotConfig:', df.query('imp_strategy=="Depends on LotConfig"').index)
    print()


#%% Fill in / impute missing values - MANDATORY

impstrategy_popular=['FireplaceQu', 'Electrical', 'Utilities','Functional','KitchenQual','Exterior2nd','Exterior1st','SaleType']
impstrategy_mean=['GarageCars','GarageArea',
                  'BsmtHalfBath','BsmtFullBath', 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
impstrategy_none=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType',
                  'GarageQual','GarageCond','GarageType','GarageFinish',
                  'BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2']
impstrategy_0=['MasVnrArea','GarageYrBlt']

for dataset in all_datasets:
    #dataset=train
    dataset[impstrategy_none] = dataset[impstrategy_none].fillna('None')
    dataset[impstrategy_0] = dataset[impstrategy_0].fillna(0)
    for i in impstrategy_popular:# could be probably rewritten with .apply()
        if i in dataset.columns:
            dataset[i]=SimpleImputer(strategy='most_frequent').fit_transform(dataset[[i]])
    for i in impstrategy_mean:
        if i in dataset.columns:
            dataset[i]=SimpleImputer(strategy='mean').fit_transform(dataset[[i]])
    dataset['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    dataset['LotFrontage'] = train.groupby(['Neighborhood','BldgType','GarageCars'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    dataset['LotFrontage'] = train.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

train.to_csv('train2.csv')
train.isnull().sum()
test.isnull().sum()

#%% Exploratory Data Analysis
# https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda

# Important variables
# - Saleprice - response: histogram
# - Top 10-20 Predictors: correlation matrix / RFE
# - Multi-colinearity
# - Strength of correlations: using whisker plots / scatter plots
# - Identification of  outliers: scatter plots

#%% Identify top features - MANDATORY
#Generate dummy variables
train_encoded_x= pd.get_dummies(train.drop('SalePrice', axis=1), drop_first=True)
train_encoded_y=train['SalePrice']
train_encoded=pd.concat([y, X.reindex(y.index)], axis=1)

# Top 20 deatures By RFE
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
selector_rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=20)
selector_rfe.fit(train_encoded_x,train_encoded_y)
best_features_rfe=train_encoded_x.columns[selector_rfe.support_]
print ('Top 20 Predictors by RFE with DecisionTreeRegressor')
print(best_features_rfe)

# Optimum feature selection using RFECV
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
time_start = perf_counter()
scoring_method='neg_mean_squared_error'
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
model = DecisionTreeRegressor()
selector_rfecv = RFECV(estimator=model, scoring=scoring_method, n_jobs=-1, cv=cv_method)
selector_rfecv.fit(train_encoded_x,train_encoded_y)
time_stop = perf_counter()
best_features_rfecv=train_encoded_x.columns[selector_rfecv.support_]
print('Best Features By RFECV with DecisionTreeClassifier')
print (len(best_features_rfecv),'out of', len(train_encoded_x.columns),'columns')
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
#   Are there any categorical values with low number of observations?
#       Groupby + clustering

#https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf

# Response: Saleprice
# Histogram / density plot - SalePrice
# SalePrice v key correlated varibles:
    # Location: Neighboroughood, MS SubZoning
    # Square Footage: TotalBsmtSF', '1stFlrSF', 2ndFlrSF' 'GrLivArea'
    # Garage: GarageCars
    # Quality: ExterQUal_TA
    # Year: YearBuilt', 'YearRemodAdd, Year Sold?
    # Lot: LotFrontage', 'LotArea

#1. SALEPRICE
#Plot histogram
plot1=sns.distplot(train['SalePrice'], color="b");
plot1.set(ylabel="Frequency")
plot1.set(xlabel="SalePrice")
plot1.set(title="SalePrice Distribution")
sns.despine(trim=True, left=True)
plt.show()

# Normality tests
stat, p = shapiro(y)
skew=train['SalePrice'].skew()
kurt=train['SalePrice'].kurt()

print('Shapiro-Wilk: stat=%.3f, p=%.3f (p>0.05 = gaussian)' % (stat, p))
print("Skewness: %f (0 = symmetrical)" % skew )
print("Kurtosis: %f (3 = normaldist)" % kurt)

#NEIGBOURGHOOD / MSSUBZONING
#plot2=sns.barplot(x='Neighborhood', y='SalePrice', data=train)
#plot2.set_xticklabels(plot2.get_xticklabels(), rotation=45)
#plot2.axhline(train['SalePrice'].mean())
#plot2.set(title="SalePrice Per Neighborhood")
# plt.show()

# Scatter plot with fill based on BldgType
#plot3=sns.scatterplot(x=train['Neighborhood'], y=train['SalePrice'], hue=train['BldgType'].tolist())
#plt.show()

plot3=sns.stripplot(x=train['Neighborhood'], y=train['SalePrice'], alpha=.25 )
plot3.set_xticklabels(plot3.get_xticklabels(), rotation=45)
plot3.axhline(train['SalePrice'].mean())
plot3.set(title="SalePrice Per Neighborhood")
plt.show()

train.dtypes

#%% Crosstab report - PROB NOT NEEDED
# Discrete Variable Correlation by Survival using
# group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
train_label_y='SalePrice'
train_labels_x=train.drop(train_label_y, axis=1).columns

for x in train_labels_x:
    if train[x].dtype not in ['float64','int64']:
        print('SalePrice Correlation by:', x)
#       print(train[[x, Target[0]]].groupby(x, as_index=False).mean())
        print(train[[x, train_label_y]].groupby(x, as_index=False).mean())
        print('-' * 10, '\n')


