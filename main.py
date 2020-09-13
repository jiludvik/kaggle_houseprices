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
from scipy.stats import shapiro, normaltest,anderson
from pingouin import multivariate_normality
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

#%% Identify top features 1- MANDATORY
# Correlation Matrix for Numerical Columns Only - OPTIONAL
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

#%% Identify top features 2- MANDATORY
#Generate dummy variables
train_encoded_x= pd.get_dummies(train.drop(['SalePrice','Id'], axis=1), drop_first=True)
train_encoded_y=train['SalePrice']
train_encoded=pd.concat([train_encoded_y, train_encoded_x.reindex(train_encoded_y.index)], axis=1)

# Top features By RFE
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
top_features_no=10
selector_rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=top_features_no)
selector_rfe.fit(train_encoded_x,train_encoded_y)
best_features_rfe=train_encoded_x.columns[selector_rfe.support_]
print ('Top', top_features_no,'Predictors by RFE with DecisionTreeRegressor')
print(best_features_rfe)

# SalePrice + Best Predictors (LotFrontage', 'OverallQual', 'YearBuilt', 'BsmtFinSF1',
# 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea)

# Optimum feature selection using RFECV
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
time_start = perf_counter()
scoring_method='neg_mean_squared_error'
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
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
#   Outliers: method tbc

#Best predictors
#    'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

# OverallQual
# YearBuilt
# Neighborhood
# LotFrontage, LotArea
# 'GarageCars', 'GarageArea

# LotArea?

#https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf

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

#1. SALEPRICE
mydistplot('SalePrice', train, y_label="Frequency", title="SalePrice Distribution", show=True)

# Scatter plot with fill color driven by a variable
#plot3=sns.scatterplot(x=train['Neighborhood'], y=train['SalePrice'], hue=train['BldgType'].tolist())
#plt.show()

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
def univar_normtest(data=None):
    norm_test_cols=['name','normal', 'shapiro_gauss','dagostino_gauss','anderson_gauss','shapiro_stat','shapiro_p','dagostino_stat','dagostino_p','andreson_crit_val','andreson_stat']
    norm_test=pd.DataFrame(columns=norm_test_cols)
    for col in data.select_dtypes(exclude='object').columns:
        shapiro_stat, shapiro_p = shapiro(data[col])
        shapiro_result=(shapiro_p>0.05)
        dagost_stat, dagost_p = normaltest(data[col])
        dagost_result = (dagost_p>0.05)
        anders = anderson(data[col])
        anders_result=(anders.statistic < anders.critical_values[2])
        result=shapiro_result or dagost_result or anders_result
        result=pd.DataFrame([[col,result, shapiro_result, dagost_result,anders_result, shapiro_stat, shapiro_p, dagost_stat, dagost_p,anders.critical_values[2], anders.statistic]], columns=norm_test_cols)
        norm_test = norm_test.append(result, ignore_index=True)
    return(norm_test[['name','normal']])

print('Univariate Normality Test:')
print(univar_normtest(train))

print('Multivariate Normality Test:')
stat, p, normal = multivariate_normality(train.select_dtypes(exclude='object').to_numpy())
print('Statistics=%.3f, p=%.3f' % (stat, p))
print('Does data set have normal distribution:', normal)

