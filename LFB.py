
import streamlit as st 
import time
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
import os 
import numpy as np
import pandas as pd
import collections
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import statsmodels.api as sm
from fancyimpute import IterativeImputer # for imputation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso # for penalised regression and feature engineering
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import random
@st.cache_data # Saves data into a cache facilitates re-run
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)

# st.title ("Analysis  of Factors affecting London  Fire Brigade Response Time.")

#pages=["Introduction to the project", "Difficulties encountered", "Mile stones"]
#page=st.sidebar.radio("Go to", pages)
st.sidebar.markdown('''
# Table of content
- [Introduction to the project](#introduction-to-the-project)
- [Difficulties encountered](#difficulties-encountered)
- [Mile stones](#mile-stones)
  - [Loading and inspection of the data](#loading-and-inspection-of-the-data)
    - [The mobilisation data](#the-mobilisation-data)
    - [The incidence data](#the-incidence-data)
    - [Merging the incidence and mobilisation data](#merging-the-incidence-and-mobilisation-data)
  - [Preprocessing](#preprocessing)
    - [Sanity checks in the mobilisation and incidence data](#sanity-checks-in-the-mobilisation-and-incidence-data)
      - [Assessment of missing values](#assessment-of-missing-values)
      - [Imputation](#imputation)
  - [Feature engineering](#modelling)
  - [Modelling](#feature-engineering)
    - [First model: Linear model](#first-model-linear-model)
    - [Second model: Random forest regression with optimal parameters](#second-model-random-forest-regression-with-optimal-parameters)
    - [Selection of significant features from hypothesis testing](#selection-of-significant-features-from-hypothesis-testing)
    - [Final model: Refitting the linear model only using significant features](#final-model-refitting-the-linear-model-only-using-significant-features)
  - [Conclusions](#conclusions)
                                                                  
''', unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: black;'>Analysis  of Factors affecting the London  Fire Brigae Response Time.</h1>", unsafe_allow_html=True)
st.markdown("## Introduction to the project")
st.markdown("<p style='text-align: justify; color: black;'>The London Fire Brigade is amongst the largest in the world and is covering a wide rage of activities like; firefighting, road traffic collision, flooding, outdoor fire,gas leak, building collapse, just to name a few. The also carryout community sensitisation and fire safety education that starts in the primary school. In this project I am going to analyse the data of the London Fire Brigade(LFB) and create a model that will be capable of predicting the attendance time . The attendance time is the time from when the first call was received at a firefighting unit till when the first machine(unit) arrives the scene of the incident.</p>", unsafe_allow_html=True)
st.markdown("## Difficulties encountered")

open_p = "<p style='text-align: justify; color: black;'>"
close_p = "</p>"
item='''The size of the dataset, made it difficult to run techniques such as 
the KNN imputation which works best on smaller datasets. The dataset consists of two 
tables, one containing mobilisation data for the LFB and the other recording information 
about the incidences.\nThe key joining both tables is the incidence number. During my analysis, 
I realised that some incidence numbers were mutually exclusive, which means only present in one of 
the datasets. This affected up to 34% of the incidence numbers. It was good that we still had almost 
70% of the data left. This means that these 34% could be discarded without worry.\nThe incidence numbers 
were not stored in the same format in both tables. In one table, the numerical parts were stored as 
integers and in the other table as floats. But the significant numbers were the same. I solved this 
by converting the integers to floats, and then stringifying the respective columns of incidence numbers 
before performing merging.After analysing both datasets, I realized a lot of missed mapping with regards 
to the "IncidentNumber" columns. This could probably be as a result of sycronisation problems caused by 
the reforms that have been taking place in the agency in the last decade.\nThere was also a problem of high rates of missing values for certain vairables. Some times as much as almost 80%. Features that had more than 14% of missing values were excluded and not imputed. Most of these variables were categorical and imputing them was likely to yield untrue data.'''

st.markdown(open_p + item + close_p, unsafe_allow_html=True)

n = 500000 # work with the first 500000 row only for now

# mile stones
st.markdown("## Mile stones")

#### data inspection ###
#st.header('Section 1')
# st.markdown("[Loading and inspection of the data](#page1)")
st.markdown("### Loading and inspection of the data")


item='''The data were loaded using the read_excel function from the pandas library. 
        The data consisted of two main datasets, the mobilisation daatset and the incidence dataset. The mobilisation data has 3 parts, collected in 
        three EXCEL files, comprising mobilisation data  spanning the years 2009-2014, 2015-2020, 2021-2024. On the other hand, the incidence data consists of two parts, one
        collected from 2009 - 2017 and 2018-2024.
        These two datasets had three common identifiers, including the incidence number, CallYear(year of the call), HourOfCall (the hour of the call).
        In total, the raw merged mobilisation data had 2358050 rows and 22 columns while the merged incidence data consisted of 1680826 rows and 39 features.'''
        
st.markdown(open_p + item + close_p, unsafe_allow_html=True)


st.markdown("#### The mobilisation data")

itemb='''As already mentioned above, the mobilisation dataset comprises of three sub-datasets collected from the year 2009 up to and including 2024. 
        They will be merged.'''

st.write(open_p + itemb + close_p, unsafe_allow_html=True)

# Reading in the first mobilisation dataset from 2009 - 2014
mobi_df1 = pd.read_excel("LFB Mobilisation data from January 2009 - 2014.xlsx", nrows=n)
# Read in the second mobilisation datset .i.e. 202....
mobi_df2 = pd.read_excel("LFB Mobilisation data from 2015 - 2020.xlsx", nrows=n)
# Reading the third mobilization dataset from the year 2021-2024
mobi_df3 = pd.read_excel("LFB Mobilisation data 2021 - 2024.xlsx", nrows=n)
# Merge the mobilisation data
full_mobi_df = pd.concat([mobi_df1, mobi_df2, mobi_df3])

st.markdown("###### Overview of the merged mobilisation data")

# st.dataframe()
# st.dataframe(full_mobi_df.head(10).style.format("{:.2%}"))


st.write(full_mobi_df.head(10).style.format({"CalYear": "{:.0f}"}))

itemb = '''Table1. After merging the mobilisation dataframes, I realise that the full data consists
of 2358050 rows and 22 columns as initially mentioned.'''

st.write(open_p + itemb + close_p, unsafe_allow_html=True)

###### Overview of the incidence data #####
st.markdown("#### The incidence data")

itemb='''The incidence dataset comprises of two sub-datasets collected from the year 2009 up to and 
        including 2024. Like the mobilisation data, they will be also merged into one
        main incidence dataset.'''

st.write(open_p + itemb + close_p, unsafe_allow_html=True)

# Loading the Incidence data from 2009-2017
incidence_df1 = pd.read_csv("LFB Incident data from 2009 - 2017.csv", nrows=n)
# Loading the Incident data from 2018 onwwards
incidence_df2 = pd.read_excel("LFB Incident data from 2018 onwards.csv.xlsx", nrows=n)
# Concatenating the two Incidence data frames
full_incidence_df = pd.concat([incidence_df1, incidence_df2])

st.markdown("###### Overview of the merged mobilisation data")

st.write(full_incidence_df.head(10).style.format({"CalYear": "{:.0f}"}))

itemb = '''Table2. After merging the mobilisation dataframes, we realise that the full data consists
of 1680826 rows and 39 features.'''

st.write(open_p + itemb + close_p, unsafe_allow_html=True)

# Summary statistics of the full mobilisation data( this method captures only the numerical columns)
full_mobi_df.describe()
st.write(full_incidence_df.head(10).style.format({"CalYear": "{:.0f}"}))

itemb = '''Summary description of the merged mobilisation dataset.'''

st.write(open_p + itemb + close_p, unsafe_allow_html=True)


#### merging the mobit and incidence data ####
st.markdown("#### Merging the incidence and mobilisation data")

itemb='''Note that there are three features that are common to the mobilisation and 
incidence datasets and these include the incidence number, CalYear(year of the call), HourOfCall (the hour of the call).
The incidence number being the most unique shared identifier,  will be used to merge the two datasets. 
However, to merge the two data frames, I had to harmonise the formats of the incidence numbers in the two
data frames. It is observable that the IncidentNumber columns in the mobilization and 
the incidence datasets are pandas.core.series. However the numerical part 
of the column in question in the incidence dataset is in float format, meanwhile, in the 
mobilization data, they are integers. To facilitate the merging of both datasets, I 
converted the incidence number to string format, since it was the best format to uniformise the 
data type of the incidence number column.After processing the IncidenceNumbers column from both datasete by 
converting them to strings and taking away some unwated charaters,I proceeded to check if 
there are duplicated incidence numbers and identify their characteristics. My assessment showed that 
almost 34% of the incidence numbers in the mobilisation dataset are duplicated. 
There were however no duplicated incidence numbers in the incidence dataset. This implied 
that I need to find a way to select only one incidence representative incidence number 
in the mobilisation dataset.\n
From observation, one could see that the same incidence number might correspond to a responses 
from two different stations, with different mobilisation statistics. 
For example they may have different response time or different logistic details. 
I will prioritise the incidence number whose data correspond to the smallest attendance time.'''



# convert incidence number to string and exclude the decimal point and the zero suffix
alist = full_incidence_df["IncidentNumber"].astype("string")
alist = [a_string.strip(".0") for a_string in alist]

# Reinserting the arranged number into the IncidentNumber column of the Incident dataset
full_incidence_df["IncidentNumber"] = alist

# same for mobilisation data, converting the IncidentNumber column to string.
full_mobi_df['IncidentNumber'] = full_mobi_df['IncidentNumber'].astype("string")

# sorting the attendance time in ascending order
full_mobi_df = full_mobi_df.sort_values(by = "AttendanceTimeSeconds")

# select first of duplicates with the smaller incidence number
full_mobi_df = full_mobi_df.drop_duplicates(subset=['IncidentNumber'], keep='first')

# Identify shared incidence numbers from the full mobilisation dataset after duplicates have been remove with full incident number.
shared_incidence_numbers = pd.Series(np.intersect1d(full_mobi_df["IncidentNumber"], 
                                                    full_incidence_df["IncidentNumber"]))

# filter mobilisation rows based on shared incidence numbers caculated above.
mask = full_mobi_df['IncidentNumber'].isin(shared_incidence_numbers)
full_mobi_subset = full_mobi_df[mask]

# Filter incidence rows based on shared_incidence_numbers
mask = full_incidence_df['IncidentNumber'].isin(shared_incidence_numbers)
full_inci_subset = full_incidence_df[mask]

# Drop the call year and hour of call from the mobilisation data to facilitate merging and avoiding duplicate of these columns
merged_data = pd.merge(full_inci_subset, full_mobi_subset.drop(['CalYear', 
                       'HourOfCall'], axis=1), on="IncidentNumber")

st.write(merged_data.head(10).style.format({"CalYear": "{:.0f}"}))

itemb = '''Table3. The final merged dataset of the mobilisation and incidence 
data cinsists of 1046976 rows and 58 features.'''

st.write(open_p + itemb + close_p, unsafe_allow_html=True)


### preprocessing ###
st.markdown("### Preprocessing")
st.markdown("#### Sanity checks in the mobilisation and incidence data")

item='''I noticed that out of the 1565807 unduplicated mobilisation dataset, only a fraction of 
them are present in the incidence dataset. Initially the Incident dataset had no duplicates and 
was made up of 1691361 rows. Here I expected a full coverage of the mobilisation dataset but this is 
not the case as we can see only 1046976 shared cases of IncidentNumbers. What accounted for this? Possibly 
a non-consistent data collection procedure over the years.\n
I also identified missing values and tested mean, median and MICE imputation approaches, 
after which I selected the best. I however could not perform KNN imputation due to the large size
of the dataset. KNN imputation best works on smaller datsets.\nA correlation analysis was performed to identify
which features correlated to each other and to what extent. This helpoed advise the feature engineering
process because a choice had to be made among highly correlated predictors.\n
Three feature engineering approaches were assessed including a linear approach such as LASSO regression,
a tree-based approach such as random forst and an approach which combines linear and non-linear approacges 
called mutual information regression.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("##### Assessment of missing values")

item='''Inspection of the data showed presence of missing values. I assessed to find out the percentage
of missing values per feature. The table below shows the percentage of missing values per feature'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Display the percentage of missing values per column of the merged_dataset
percent_missing = merged_data.isnull().sum() * 100 / len(merged_data)
percent_missing.sort_values(ascending=False)

# create a data frame of features and their percentages of missing values
percent_missing_df = pd.DataFrame(percent_missing.tolist(), index = percent_missing.index)

# name the columns of the data frame of missing values
percent_missing_df.columns = ['Proportions_NANs']

percent_missing_df.index.name = 'Feature'
percent_missing_df.reset_index(inplace=True)

percent_missing_df.head(20)

# identify features with percentage if missing values > 0
perc_with_NAs = percent_missing_df[percent_missing_df.Proportions_NANs > 0]
perc_with_NAs_sorted = perc_with_NAs.sort_values('Proportions_NANs', ascending=False)
perc_with_NAs_sorted

item='''Table4. Percetanges of missing values per feature.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)


item='''The bar plot below shows a vosualisation of features containing missing values and 
the percentage of these missing values per feature.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# perc_with_NAs.sort_values(ascending=False, by="Proportions_NANs").plot.bar(perc_with_NAs.Feature,figsize = (10,5))

plt.bar(perc_with_NAs_sorted.Feature, perc_with_NAs_sorted.Proportions_NANs)
plt.xticks(rotation=45, ha='right');
plt.title('Plot of Features and their Percentage of Missing Values')
plt.xlabel('Feature')
plt.ylabel('Percentage of Missing Values')
plt.axhline(y=5, color='r', linestyle='--')
plt.xticks(rotation=45, ha='right')
st.pyplot(plt.gcf())


item='''Comment: It is so far unclear how features with very high rates of missing 
values relate to the outcome variable of interest. But at least for legal reasons, 
it will be important for the LFB to enhance data collection and recording for features having >= 15% missing value rates..'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("##### Imputation")

item='''Imputation is a technique that can help us replace missing nunbers by intelligently guessing 
new ones to replace them. However imputation is best practised when the percentage of missing data is 
not greater than 5% and also if the data is numeric. For this reason we are excluding all features with 
more than 10% of missing values.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# exclude columns with more than 10% NAs
limitPer = len(merged_data) * .90
merged_data = merged_data.dropna(thresh=limitPer, axis=1)
percent_missing = merged_data.isnull().sum() * 100 / len(merged_data)

# select those with lowest missing value rates.
with_missing_values = percent_missing [percent_missing > 0]

item='''Table5. The table above contains features that have percentages of missing values that
are low enough to permit accurate imputation.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

item='''I perform imputation on the missing data using a variety of techniques including mean, median 
and MICE imputation. However this can be best done on numeric data. So categorical data that have 
missing values will not be subjected to imputation. Also I tried KNN imputation, but the algorithm 
failed to work because of the larg size of the dataset. Please note that due to the structure of the dataset
and its constituent features, imputation was only performed on imputable numerical features.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# selecting the numerical columns
missing_numeric_columns = merged_data.select_dtypes(include=np.number).columns.tolist()

# the ResourceMobilisationId column is numerical but categorical and not helpful in this case. So I exclude it
merged_na_data_numeric = merged_data[missing_numeric_columns].drop(['ResourceMobilisationId', 'PumpOrder'], axis="columns")
# merged_na_data_numeric['PumpOrder'] = merged_na_data_numeric['PumpOrder'].astype('category')
# merged_na_data_n

### MICE imputation ###
#### mice imputation ####

st.markdown("**Imputation by Multivariate Imputation by Chained Equations (MICE)**")

item='''The process uses multiple imputation techniques to fill in the missing data 
and then combines the results from multiple imputations to produce a final imputed dataset.\nAfter performing the
MICE imputation, I compared the mean of the features after the imputaitonsto the mean 
of the same columns before imputation and I observed very little difference. I also performed a correlation
between the respectuve features before and after correlation and observed always a correlation coefficient > 0.9.\n
I also performed a linear regression analysis with the outcome variable of interest using the imputed data. I subsequently 
obtain an r-squared value which will enable me assess the suitability 
of this imputation method compared to others.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Copy the data to merged_mice_merged_na_data_numeric_imputed
merged_mice_merged_na_data_numeric_imputed = merged_na_data_numeric.copy(deep=True)

# Initialize IterativeImputer
imputer = IterativeImputer()

# Impute using fit_tranform on merged_mice_merged_na_data_numeric_imputed
merged_mice_merged_na_data_numeric_imputed.iloc[:, :] = imputer.fit_transform(merged_mice_merged_na_data_numeric_imputed)
# check the mean
merged_mice_merged_na_data_numeric_imputed.loc[:, 'TravelTimeSeconds'].mean()
# check the original mean
merged_data.loc[:, 'TravelTimeSeconds'].mean()
# correlate imputed value of TravelTimeSeconds with the AttendanceTimeSeconds
merged_mice_merged_na_data_numeric_imputed['TravelTimeSeconds'].corr(merged_data['AttendanceTimeSeconds'])

# perform a linear regression to estimate R-squared value when we regress the outcome of interest and imputed predictors
merged_mice_merged_na_data_numeric_imputed["AttendanceTimeSeconds"] = merged_data["AttendanceTimeSeconds"]
X = sm.add_constant(merged_mice_merged_na_data_numeric_imputed.drop("AttendanceTimeSeconds", axis='columns'))
y = merged_mice_merged_na_data_numeric_imputed['AttendanceTimeSeconds']
lm_MICE = sm.OLS(y, X).fit()

#### Mean imputation ####

st.markdown("**Imputation by mean**")

item='''The imputation by mean method replaces each missing value with the mean value for that feature.
One set back with this method is that variations in the data are missed out.\nAfter performing the
mean imputation, I compared the mean of the features after the imputaitonsto the mean 
of the same columns before imputation and I observed very little difference. I also performed a correlation
between the respectuve features before and after correlation and observed always a correlation coefficient > 0.9.\n
I also performed a linear regression analysis with the outcome variable of interest using the imputed data. I subsequently 
obtain an r-squared value which will enable me assess the suitability 
of this imputation method compared to others.\n
Considering that imputation by mean does not capture the variation present in the data, I expected 
a much weaker correlation between the imputed data and the data before imputation. I also expected a greater 
difference between the means before and after imputation. The fact that the difference between the features 
before and after mean imputation show that the percentages of NAs were so low that mean imputation
did not distor the overall structure of the data.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# mean imputation
from sklearn.impute import SimpleImputer

# Make a copy of merged_na_data_numeric
merged_na_data_numeric_mean = merged_na_data_numeric.copy(deep=True)

# Create mean imputer object
mean_imputer = SimpleImputer(strategy='mean')

# Impute mean values in the DataFrame merged_na_data_numeric
merged_na_data_numeric_mean.iloc[:, :] = mean_imputer.fit_transform(merged_na_data_numeric_mean)

# check mean
merged_na_data_numeric_mean.loc[:, 'TravelTimeSeconds'].mean()

# mean original
merged_data.loc[:, 'TravelTimeSeconds'].mean()

# perform a linear regression to estimate R-squared value when we regress the outcome of interest and imputed predictors
merged_na_data_numeric_mean["AttendanceTimeSeconds"] = merged_data["AttendanceTimeSeconds"]
X = sm.add_constant(merged_na_data_numeric_mean.drop("AttendanceTimeSeconds", axis='columns'))
y = merged_na_data_numeric_mean['AttendanceTimeSeconds']
lm_mean = sm.OLS(y, X).fit()


### Median imputation ###

st.markdown("**Imputation by median**")

item='''The imputation by median method replaces each missing value with the median value for that feature.
One set back with this method is that variations in the data are missed out.\nAfter performing the
median imputation, I compared the median of the features after the imputaitonsto the median 
of the same columns before imputation and I observed very little difference. I also performed a correlation
between the respectuve features before and after correlation and observed always a correlation coefficient > 0.9.\n
I also performed a linear regression analysis with the outcome variable of interest using the imputed data. I subsequently 
obtain an r-squared value which will enable me assess the suitability 
of this imputation method compared to others.\n
Considering that imputation by median does not capture the variation present in the data, I expected 
a much weaker correlation between the imputed data and the data before imputation. I also expected a greater 
difference between the medians before and after imputation. The fact that the difference between the features 
before and after median imputation show that the percentages of NAs were so low that median imputation
did not distor the overall structure of the data.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Make a copy of merged_na_data_numeric
merged_na_data_numeric_median = merged_na_data_numeric.copy(deep=True)

# Create mode imputer object
median_imputer = SimpleImputer(strategy="median")

# Impute using most frequent value in the DataFrame median_imputer
merged_na_data_numeric_median.iloc[:, :] = median_imputer.fit_transform(merged_na_data_numeric_median)

# perform a linear regression to estimate R-squared value when we regress the outcome of interest and imputed predictors
merged_na_data_numeric_median["AttendanceTimeSeconds"] = merged_data["AttendanceTimeSeconds"]
X = sm.add_constant(merged_na_data_numeric_median.drop("AttendanceTimeSeconds", axis='columns'))
y = merged_na_data_numeric_median['AttendanceTimeSeconds']
lm_median = sm.OLS(y, X).fit()

### Assessment of imputation approaches ###
st.markdown("**Which imputation approach is better?**")
# Store the Adj. R-squared scores of the linear models
rsquared_df = pd.DataFrame({'Mean Imputation': lm_mean.rsquared_adj, 
                            'Median Imputation': lm_median.rsquared_adj, 
                            'MICE Imputation': lm_MICE.rsquared_adj}, 
                         index=['Adj. R-squared'])

# Neatly print the Adj. R-squared scores in the console
rsquared_df

st.write(rsquared_df.style.format({"Mean Imputation": "{:.0f}"}))

item='''The table above shows that MICE imputation leads to predictors that best 
explain the outcome of interest. This is because it has the highest R-squared value. 
This implies that a linear regression constructed from MICE-imputed values, almost 
perfectöy explain 100% of the variation in the mode.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)


merged_data_cc = merged_data.dropna(how="any")
merged_data_cc

# Plot graphs of imputed DataFrames and the complete case
merged_data_cc['AttendanceTimeSeconds'].plot(kind='kde', c='red', linewidth=3)
merged_mice_merged_na_data_numeric_imputed['AttendanceTimeSeconds'].plot(kind='kde')
merged_na_data_numeric_median['AttendanceTimeSeconds'].plot(kind='kde')
merged_na_data_numeric_mean['AttendanceTimeSeconds'].plot(kind='kde')

# Create labels for the four DataFrames
labels = ['Baseline (Complete Case)', 'Mean Imputation','Median Imputation', 'MICE Imputation']
plt.cla()
plt.clf()
plt.legend(labels)

plt.title('Comparison of imputation approaches.')
# Set the x-label as Skin Fold
plt.xlabel('AttendanceTimeSeconds')

# plt.show()
st.pyplot(plt.gcf())
plt.clf()  
item='''Figure2. Comparison of different imputation approaches.'''

item='''We would have expected different distributions of the data imputed from MICE, 
mean and median imputations. They however appear close here. Theoretically, MICE imputation is 
the best of the three since it also takes in to account correlations with other variables, 
which is a feature absent in mean or median approach. We also see that it has the highest R-squared. 
The graph above also shows that it is also bimodal, just like the baseline distribution. The mean and 
median imputations are not catastrophically bad as expected, prpbaboly because of the relatively low 
percentages of missing data, which makes it in a way that the disadvantages of mean and median 
imputation are not very prominent. On the other hand, it could also show that variations are not much 
such that mean and median values do not deviate much from what is currently existing.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)


### correlation between features ###

st.markdown("**Correlation between features.**")


item='''The plot below contains a Pearson correlation analysis between features. It will be  
a first step towards assessing auto-correlation between predictor variables.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# correlation of MICE imputed data
corr = merged_mice_merged_na_data_numeric_imputed.corr(method='pearson')
plt.cla()
plt.clf()
ax0 = plt.axes()
ax0.set_title('Pearson correlation between MICE imputed variables in the daset')
sns.heatmap(corr, cmap='RdBu_r', vmin = -1, vmax = 1)
# plt.show()
st.pyplot(plt.gcf())


item='''From the plot above, we can already suspect autocorrelation between predictor variables. 
This can affect model performance. It will be handled later. We also see that certain values best 
correlate with the outcome of interest.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

mice_imp_columns = merged_mice_merged_na_data_numeric_imputed.columns
merged_data[mice_imp_columns] = merged_mice_merged_na_data_numeric_imputed[mice_imp_columns]

# exclude this column DateAndTimeMobile
merged_data2 = merged_data.drop(['DateAndTimeMobile', 'DateAndTimeMobile',
       'DateAndTimeMobilised', 'TimeOfCall', 'PumpOrder'], axis='columns')
merged_data2 = merged_data2.select_dtypes(include=np.number).drop(["ResourceMobilisationId"], axis = "columns")

st.markdown("#### Feature engineering")

item='''I want to identify using lasso regression and random forest regression  
features that are best predictive of the outcome of interest which is the AttendanceTimeSeconds.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("**Standardisation of features**")

item='''The goal of standardisation is to remove the effect of scale in various variables.
Algorithms tend to bias in favour of larger numbers. By standardising, the mean of each feature
becomes zero and the standard deviation 1. Standardisation gives equal weiting to the variables irrespective
of scale.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

#X and y values
X = merged_data2.drop("AttendanceTimeSeconds", axis=1)
y = merged_data2.AttendanceTimeSeconds

#Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

st.markdown("**Train-test split for cross validation**")

item='''To perform cross-validation, I performed the train-test split such that
70% of the data are in the training dataset and the remaining 30% in the test dataset.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

features = X.columns
#splot
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=17)


st.markdown("**Lasso regression**")

#Lasso regression model
print("\nLasso Model............................................\n")

#Lasso Cross validation
lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], random_state=17).fit(X_train, y_train)

#score
print(lasso_cv.score(X_train, y_train))
print(lasso_cv.score(X_test, y_test))

print("The train score for ls model is {}".format(lasso_cv.score(X_train, y_train)))
print("The test score for ls model is {}".format(lasso_cv.score(X_test, y_test)))

plt.cla()
plt.clf()
pd.Series(lasso_cv.coef_, features).sort_values(ascending = True).plot(kind = "bar",
          title = "Feature selection by Lasso regression")
st.pyplot(plt.gcf())

item='''Lasso regression is a strict feature engineering 
approach. It excludes all features that have very low or insignificant predictive power. It only identiy 
three features that are predictive of the outcome. The travel time in seconds has the highest 
predictive value'''

st.write(open_p + item + close_p, unsafe_allow_html=True)


st.markdown("**Random forest regression**")

item='''I perform an analysis of feature importance using random forest regression. I perform a train-test
split where 70% of the data are retained in the training dataset and the remaining 30% in the 
testing dataset.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

#X and y values
X = merged_data2.drop("AttendanceTimeSeconds", axis=1)
y = merged_data2.AttendanceTimeSeconds

features = X.columns

#splot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_jobs = 8,
                                 random_state=0,
                                 n_iter=5, return_train_score = True,
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

import pandas as pd
df = pd.DataFrame(rand_search.cv_results_)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, 
                                index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
plt.cla()
plt.clf()
feature_importances.plot.bar(title = "Assessment of feature importance by random forest regression")
st.pyplot(plt.gcf())

item='''Figure4.Lasso Regression and Random Forest Regressions have been able to capture the 
best first three parameters but in different orders. It should be noted that the three parameters 
identified with the random forest as having the highest predictive importance on the 
AttendanceTimeSeconds are the same as those lasso identified as being of predictive value for the 
AttendanceTimeSeconds.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

#### mutual information regression ####
st.markdown("**Mutual information regression**")

item='''This helps us identify top k features that are best predictive of the outcome, 
using mutual information. This approach can pick up both linear and non-linear relationships.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from functools import partial

# Instantiating the score function
score_func = partial(mutual_info_regression, random_state=68)

# Select top 10 features with the most mutual information
selection = SelectKBest(score_func=score_func, k=10)

print(selection.fit_transform(X, y))

X_use = X[X.columns[selection.get_support(indices=True)]]

print(X_use)
   
               
# Display of correlation between the top 10
sns.heatmap(X_use.corr(), cmap='RdBu_r')
plt.title('Display of correlation between the top 10 predictive features')
# plt.show()
st.pyplot(plt.gcf())

item='''Figure5. Correlation between features with best mutual information with attendance time'''

st.write(open_p + item + close_p, unsafe_allow_html=True)


item='''Random forest can pick up both linear and non-linear relationships between predictors 
and the outcome. Linear approaches such as lasso can only identify the linear relationships. 
Mutual information can identify both. I rather priorise the top k = 10 features identify by 
mutual information regression so we have more featurs in our model. Through trial and error, 
we saw that taking more than 10 predictirs, led to issues related to autocorrelation between 
the variables. Taking 10 is a solution because, we are selecting the variables that are best 
related to the outcome. This makes mutual information a variable selection approach. Neither 
random forest not lasso regression are feature selection approaches as such.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)
     

st.markdown("### Modelling")

st.markdown("#### First  model: Linear model")

st.markdown("**Standadisation of predictors**")

item='''Predictors were standardised to give them the values equal influence'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

from scipy import stats 
X_use_scaled = stats.zscore(X_use)
st.write(X_use_scaled.describe().style.format({"CalYear": "{:.0f}"}))

item='''Table6. We can see that the mean of each column is almost zero and standard deviation 1.This
schows success of standardisation process.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("**Train-test splitting**")

item='''I perform a 70% to 30% train test split, with a random state value of 6.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Splitting the data in to train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X_use_scaled, 
                                                    y, train_size = 0.7, test_size = 0.3, random_state=6)

# Create the model
reg_model = LinearRegression().fit(x_train, y_train)

st.markdown("**Linear model testing**")
# Testing the first model
y_predict = reg_model.predict(x_test)
plt.cla()
plt.clf()
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("AttendanceTimeSec: $Y_i$")
plt.ylabel("Predicted AttendanceTimeSec: $\hat{Y}_i$")
plt.title("Actual vs Predicted AttendanceTimeSec")
st.pyplot(plt.gcf())

item='''Figure6. The relationship between the predicted and actual values is a straight line
showing an almost perfect relationship.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

residuals = y_predict - y_test

plt.cla()
plt.clf()
plt.scatter(y_predict, residuals, alpha=0.4)
plt.title('Residual Analysis')
st.pyplot(plt.gcf())

item='''Figure7. We can identify unsusual patterns in the residual vs fitted plot. 
Normally we will expect that there is no visible relationship between the distribution of the
 residuals and the fitted values. The patterns we see on the plot do not call for alarm since many 
 of the predictors had values that were repeated among the observations. e.g. call year, call hour.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("**Perform linear regression again for hypothesis testing, to see how each predictor contributes to the attendance time**")
import statsmodels.api as sm
x_train_lm = sm.add_constant(x_train)
lm = sm.OLS(y_train, x_train_lm).fit()
st.write(lm.summary())

item='''The results of lm, show that the intercept, CalYear, TravelTimeSeconds, TurnoutTimeSeconds and 
FirstPumpArriving_AttendanceTime, are significant. In our context, the intercept represents that average 
attendance time at average values of all the input parameters. This is because we have standardised the 
input variables. This means that at average values of all the input variables, we have an attendance 
time of 320.44s, which is the coefficient of the intercept. The intercept in our case is hence 
meaningful, and this is not always the case\n.

It should be noted that among the significant variables, only the intercept, the turnout time, and 
travel time substantially contribute to the attendance time. This is because these features have higher 
estimates. The rest of the parameters have a minor contribution to the attendance time as their 
coefficients are close to zero. This means that as these predictors increase by one unit, there is a 
negligible increase in attendance time.\n
The coefficient of the turnout time can be interpreted as such: When the turnout time 
increases by one second, the attendance time increases by 53 seconds on average. This 
interpretation applies to all the predictors.
'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("**Mean squared error estimation**")
from numpy import sqrt 
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(y_test, y_predict))

item='''The root mean squared error using the first linear model is 0.52'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("#### Second model: Random forest regression with optimal parameters.")

item='''Perform rf again using optimised values, to see how 
each predictor contributes to the attendance time.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Instantiate rf with optimal parameters
rf = RandomForestRegressor(random_state=0, max_depth=best_rf.max_depth, n_estimators=best_rf.n_estimators, n_jobs = -1)
rf.fit(x_train, y_train)

st.markdown("**Random forest regression testing.**")
item='''Calculate the root mean squared error (rmse) of the rf.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

y_pred = rf.predict(x_test)
rms = sqrt(mean_squared_error(y_test, y_pred))

item='''The root mean squared error using the rndom forest model is 6.06. 
Comparing the RMSE of the linear model with that of random forest, I see that lm 
performs better than rf. This is because the rmse of the linear regression is far 
lower than the rmse of the random forest.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

st.markdown("#### Selection of significant features from hypothesis testing.")

item='''Final model only using significnat features.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Filter only those with a P-value less than 5% - this will be a pandas series
keep_pvals= lm.pvalues[lm.pvalues < .05]

# create a data frame of features and their percentages of missing values
keep_pvals = pd.DataFrame(keep_pvals.tolist(), index = keep_pvals.index)

# name the columns of the data frame of missing values
keep_pvals.columns = ['Pvalues']
keep_pvals

final_features = keep_pvals.index.to_list()[1: ]
final_features

st.markdown("#### Final model: Refitting the linear model only using significant features.")
# Linear model only with significant features.
x_lm = sm.add_constant(X[final_features])
lm_final = sm.OLS(y, x_lm).fit()
y_pred = lm_final.predict()

plt.cla()
plt.clf()
plt.scatter(y, y_pred, alpha=0.4)
plt.xlabel("AttendanceTimeSec: $Y_i$")
plt.ylabel("Predicted AttendanceTimeSec: $\hat{Y}_i$")
plt.title("Actual vs Predicted AttendanceTimeSec")
st.pyplot(plt.gcf())

st.markdown("**Evaluation of the final linear model.**")

# Analysis of the Residual plot.
residuals = y_pred - y

plt.cla()
plt.clf()
plt.scatter(y_pred, residuals, alpha=0.4)
plt.title('Residual Analysis')

st.pyplot(plt.gcf())

st.markdown("**Root mean squared error estimation.**")
rms = sqrt(mean_squared_error(y, y_pred))

item='''As we can se the value is very small campare to those previously calculated above.'''

st.write(open_p + item + close_p, unsafe_allow_html=True)

# Observing the correlation between the highly significant features and the dependent variable.
final_features.append("AttendanceTimeSeconds")

st.markdown("**Correlation plot.**")

# Display of the correlation plot(Heatmap)
plt.cla()
plt.clf()
sns.heatmap(merged_data2[final_features].corr(), cmap='RdBu_r')
plt.title('Display of correlation between thesignificant predictive features')
st.pyplot(plt.gcf())

st.markdown("#### Conclusions")

item='''I have used several complementary approaches to identify features or parameters 
that best predict the attendance time. These include CalYear, 'FirstPumpArriving_AttendanceTim, 
'TurnoutTimeSecon and 'TravelTimeSecc. Considering limited resources, it might not be cost-effective 
to simultaneously improve performance in all these parameters. However, the linear regression shows 
that the two outstanding contributors are the turnout time and even more importantly the travel time. 
The travel time in turn has a correlation with the attendance time of the first arriving pump. 
The travel time can hence be prioritised for action. Mechanisms can be put in place to enable LFB 
to deploy to incident zones on time. Money could also be invested to develop drones that can help 
fight fires.nds..'''

st.write(open_p + item + close_p, unsafe_allow_html=True)