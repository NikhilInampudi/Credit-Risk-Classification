# Credit Risk Classification Project
![Credit Risk Picture](https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/50634a931067490d3a5fe12af0637f24ac896de1/Credit%20Risk%20Image.jpg)

## Table of Contents 📖
- [Overview](#overview)
- [Tools Used](#tools-used)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Data Collection and Cleaning](#data-collection-and-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Pre-Processing](#pre-processing)
- [Model-Building](#model-building)
- [Model-Evaluation](#model-evaluation)
- [Findings](#findings)
- [Future Work and Improvements](#future-work-and-improvements)
- [Learnings](#learnings)

## Overview
The credit risk classification project focuses on developing a machine learning model to assess the likelihood of borrowers defaulting on 
loans, a critical challenge in the finance industry. By leveraging historical financial data, the project aims to classify borrowers into risk 
categories, enabling lenders to make informed decisions and mitigate potential losses. My interest in this project stems from a passion for 
solving complex problems in finance, particularly in doing data analysis to generate insights and predictive modeling to uncover future trends. The project not 
only aligns with my goal of applying data science to real-world financial challenges but also offers an opportunity to explore innovative solutions 
that can improve credit risk management. The project is implemented in a Jupyter Notebook and leverages Python libraries such as pandas, matplotlib, 
seaborn, plotly, sci-kit learn, and xgboost for data processing, manipulation, visualization, and machine learning.

**Dataset Link**
- [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## Full project in Jupyter Notebook [here](https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/55fa2a9826c4aa0cc9c1b5d1b4c4437e92065aa7/Credit%20Risk%20Classification%20Project.ipynb)!

## Tools Used 
- Visual Studio Code
- Jupyter Notebook
- Python Libaries:
  - Pandas
  - Matplotlib
  - Seaborn
  - Plotly
  - Sci-kit Learn
  - XGBoost

## Problem Statement
The finance industry struggles with accurate credit risk assessment due to reliance on limited or outdated criteria, resulting in suboptimal lending decisions. This project seeks to enhance risk evaluation by leveraging advanced techniques to improve decision-making, lower default rates, and identify key factors influencing a borrower's creditworthiness.

## Project Structure
The project is divided into several sections, each focusing on a different aspect of data science:

1. **Data Collection/Cleaning:** Reading historical credit data into dataframe and cleaning to make sure dataset is suitable for initial analysis.

2. **Exploratory Data Analysis (EDA):** Performing data analysis to understand the data distribution, identify trends, and generate insights. Creating visual representations of the data to better understand relationships between variables. 

4. **Pre-Processing:** Performing feature engineering, feature selection, binning, encoding and scaling relevant features to optimize classification outcomes and ensure data consistency. 
   
5. **Model Building:** Applying algorithms such as logistic regression, random forest classifier, and xgboost to train and test the data. 

6. **Model Evaluation:** Comparing metrics between models and assessing which one to use for future use. 


## Data Collection/Cleaning 
The project starts by reading the data and getting it ready so we can do some analysis. The key steps in this stage include:

**Adding necessary dependencies and reading data into dataframe**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Importing project dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\nikhi\OneDrive\Desktop\credit_risk_dataset.csv")

df.head()
```
<br><br>
**Dropping rows with null values and validating they were dropped**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Checking null value counts for all columns
df.isnull().sum()

#Dropping all rows that have null values and saving to original dataframe
df.dropna(axis=0, inplace=True)

df.isnull().sum()
```
<br><br>
**Understanding class balance/imbalance for our target variable**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Checking value count for target variable to see class balance/imbalance
df['loan_status'].value_counts()
```
<br><br>
*Initial data cleaning is short as data came in a mostly structured format. Most of the data cleaning/transformation will be performed during the pre-processing stage*

## Exploratory Data Analysis 
Exploratory Data Analysis (EDA) is a critical step in the data analysis process. It involves investigating and summarizing the main characteristics of a dataset, often using visual methods, to understand its structure, patterns, and relationships before applying more formal statistical techniques or machine learning models. In this phase, I used different visualization techniques and methods to identify correlations between variables, visualize distributions, and assess critical financial features.

**Using line plot to check how age compares to income. This shows as there is no linear relationship until the person is 120 years or older.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Plotting to see if any relationship between age and income
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x='person_age', y='person_income', data=df, ax=ax)
plt.title('Income vs Age')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/6295c4ebff1a42b9072944e8c214d05c7cd46e8b/Visualizations/Income%20vs%20Age%20Visual.png" width="900" height="400" />

<br><br>
**Using line plot to check how employment length compares to income. This shows us there are fluctuations until someone has been employed for longer than 40 months.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x='person_emp_length', y='person_income', data=df, ax=ax)
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/6295c4ebff1a42b9072944e8c214d05c7cd46e8b/Visualizations/Employment%20Length%20vs%20Income%20Visual.png" width="900" height="400" />

<br><br>
**Using line plot to check how loan amount compares to income. This shows us there are is no real linear relationship between the two.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Plotting to see if any relationship between loan amount and income
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x='loan_amnt', y='person_income', data=df, ax=ax)
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/6295c4ebff1a42b9072944e8c214d05c7cd46e8b/Visualizations/Loan%20Amount%20vs%20Income%20Visual.png" width="900" height="400" />

<br><br>
**Using correlation heatmap to see all my numerical variables relate to one another. As seen here, multiple variables such as credit history length/age and loan percent income/loan amount have a high correlation with one another. These may need to be dealt with by merging or dropping to prevent multicollinearity.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Using seaborn heatmap for better correlation visualization
fig, ax = plt.subplots() 
fig.set_size_inches(15,8)
sns.heatmap(df_numeric.corr(), vmax =.8, square = True, annot = True,cmap='Blues' )
plt.title('Confusion Matrix',fontsize=15);
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/aa7dc62b43af4a560e4bd42875242eb4dfb4a2dc/Visualizations/Credit%20Risk%20Confusion%20Matrix.png" width="900" height="700" />

<br><br>
**Plotly pie chart to visualize distribution of home ownership. This tells us an overwhelming majority of people are stil paying for their living situation. This is often a huge expense for someone to pay for and could have an effect on someones likelihood of defaulting on a loan as they have less leftover income.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Plotly pie chart to see distribution of home ownership 
import plotly.express as px

home_counts = df.person_home_ownership.value_counts()

fig=px.pie(values = home_counts.values,
           names = home_counts.index,
           color_discrete_sequence=px.colors.sequential.Mint,
           title = 'Home Ownership Distribution'
           )
fig.update_traces(textinfo='label+percent+value', textfont_size=13,
                  marker=dict(line=dict(color='#102000', width=0.2)))

fig.update_layout(width=800, height=600)

fig.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Home%20Ownership%20Pie%20Chart.png" width="900" height="650" />

<br><br>
**Plotly pie chart to visualize distribution of different loan grades. Based on this results we can see that around 80% of individuals have a satisfactory rating (A-C) and a mminority of individuals have a bad rating (D-G). This would probably have a big effect on credit worthinesss as this is going off of someones prior history of using credit.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Plotly pie chart to see distribution of loan grades
import plotly.express as px

loan_grade_counts = df.loan_grade.value_counts()

fig=px.pie(values = loan_grade_counts.values,
           names = loan_grade_counts.index,
           color_discrete_sequence=px.colors.sequential.Mint,
           title = 'Loan Grades Distribution'
           )
fig.update_traces(textinfo='label+percent+value', textfont_size=13,
                  marker=dict(line=dict(color='#102000', width=0.2)))

fig.update_layout(width=800, height=600)

fig.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Loan%20Grade%20Pie%20Chart.png" width="900" height="650" />

<br><br>
**Plotly histogram chart to visualize distribution of different loan intents. This distribution is uniform and shows us that there are multiple probable reasons for trying to get a loan**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Plotly pie chart to see distribution of loan grades
import plotly.express as px

loan_grade_counts = df.loan_grade.value_counts()

fig=px.pie(values = loan_grade_counts.values,
           names = loan_grade_counts.index,
           color_discrete_sequence=px.colors.sequential.Mint,
           title = 'Loan Grades Distribution'
           )
fig.update_traces(textinfo='label+percent+value', textfont_size=13,
                  marker=dict(line=dict(color='#102000', width=0.2)))

fig.update_layout(width=800, height=600)

fig.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Loan%20Intent%20Bar%20Chart.png" width="1300" height="500" />

<br><br>
**Plotly histogram to visualize count distribution of credit history lengths. This shows us that the data is right skewed and an overwhelming majority of individuals have had credit for only 0-5 years with a moderate amount having credit for 5-10 years. 

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Plotly histogram to see distribution of credit history lengths
fig=px.histogram(df, x = 'cb_person_cred_hist_length',histnorm = 'density', 
                 text_auto = '.2f',template = 'presentation', title = 'Credit History Length Distribution',
                 color_discrete_sequence=px.colors.sequential.Mint)
fig.update_layout()

fig.update_layout(width=1000, height=650)

fig.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Credit%20History%20Length%20Density.png" width="1100" height="650" />

<br><br>
**Seaborn pairplot to see how all the variables relate to one another with the data points color separated by their loan status. This gives us insight into whether someone defaulted on their loan based around a certain metric.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Using seaborn pairplot to see relationships between all variables and identifying which data points are loan risks
sns.pairplot(df, hue='loan_status')
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Credit%20Risk%20Pairplot.png" width="1100" height="800" />

<br><br>
**Visualizing age count of individuals in the dataset. Most of the ages are beween 20-60 and anyone after age 80 can be considered an outlier**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Using matplotlib bar chart to visualize count distribution for persons age
vertical = df['person_age'].value_counts().values
horizontal = df['person_age'].value_counts().index

plt.figure(figsize=(15, 6))

plt.bar(horizontal, vertical)
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Age%20Count%20Distribution.png" width="1500" height="600" />

<br><br>
**Visualizing employment length count of individuals in the dataset. Most of the employment lengths fall between 0-25 and anyone with employment length 40+ can be considered an outlier**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Using matplotlib bar chart to visualize count distribution for persons employment length
vertical = df['person_emp_length'].value_counts().values
horizontal = df['person_emp_length'].value_counts().index

plt.figure(figsize=(15, 6))

plt.bar(horizontal, vertical)
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/506e59afda2732565c70022b3dbf90cbe59ab003/Visualizations/Employment%20Length%20Count%20Distribution.png" width="1500" height="600" />

## Pre-Processing 
Preprocessing is essential in credit risk classification to ensure data quality and improve model performance. It addresses issues like missing values, noise, and outliers, which can skew results, by imputing or removing problematic data. Additionally, preprocessing involves feature engineering, where relevant features (e.g., debt-to-income ratio) are selected or created, and transformations (e.g., scaling, encoding categorical variables) are applied to make the data suitable for machine learning algorithms. This step ensures the dataset is clean, consistent, and optimized for accurate credit risk prediction.

<br><br>
### Addressing Outliers

**Identifying outlier data points using box and whisker plot. In this case I drop the extreme ages that exist past 80 years old.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
import numpy as np


plt.boxplot(df['person_age'], flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none'))

mean = np.mean(df['person_age'])
std_dev = np.std(df['person_age'])

plt.axhline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')

plt.legend()
plt.title('Age Distribution')
plt.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/1a0732ac7637846c672f906d6e265369bbc09a1c/Visualizations/Age%20Boxplot.png" width="900" height="800" />

<br><br>
**Identifying outlier data points using box and whisker plot. In this case I drop the extreme employee lengths that exist past 40.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
import numpy as np


plt.boxplot(df['person_emp_length'], flierprops=dict(marker='o', markerfacecolor='green', markersize=8, linestyle='none'))

mean = np.mean(df['person_emp_length'])
std_dev = np.std(df['person_emp_length'])

plt.axhline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')

plt.legend()
plt.title('Employee Length Distribution')
plt.show()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/1a0732ac7637846c672f906d6e265369bbc09a1c/Visualizations/Employee%20Length%20Boxplot.png" width="900" height="800" />

<br><br>
### Feature Engineering
**Binning numerical column "ages" into discrete categories "age group" to simplify data and make it easier to analyze**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Categorizing ages into age groups
df['age_group'] = pd.cut(df['person_age'],
                           bins=[20, 26, 36, 46, 56, 66],
                           labels=['20-25', '26-35', '36-45', '46-55', '56-65'])
```

<br><br>
**Binning numerical column "person income" into discrete categories "income group" to simplify data and make it easier to analyze**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Categorizing incomes into income groups
df['income_group'] = pd.cut(df['person_income'],
                           bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                           labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
```

<br><br>
**Binning numerical column "loan amount" into discrete categories "loan amount group" to simplify data and make it easier to analyze**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Categorizing loan amounts into loan amount groups
df['loan_amount_group'] = pd.cut(df['loan_amnt'],
                           bins=[0, 5000, 10000, 15000, float('inf')],
                           labels=['small', 'medium', 'large', 'very large'])
```

<br><br>
**Combining features to reduce dimensionality. Will assess if the features provide any value during model evaluation stage by using feature importance**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Creating new columns out of existing columns
df['emp_length_to_loan_amnt_ratio'] = df['person_emp_length'] / df['loan_amnt']

df['int_rate_to_loan_amnt_ratio'] = df['loan_int_rate'] / df['loan_amnt']
```

<br><br>
**Encoding categorical variables into numbers so they can be read by machine learning algorithms and the chi square test. This is the ideal encoding method for ordinal variables but can be insufficient for nominal variables as an algorithm might try to derive an order based on the numbers. I opted for this method because I didnt want to create multiple other columns through one hot encoding and make the dataframe more cluttered.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Label encoding categorical variables
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.columns:
    if col in df[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'age_group', 'income_group', 'loan_amount_group']]:
        df[col] = encoder.fit_transform(df[col])
```

<br><br>
**Using a for loop to scale all numerical columns ensures that the features are transformed into a consistent range, enabling better interpretation and performance by the machine learning algorithm. This process helps standardize the data, preventing features with larger magnitudes from disproportionately influencing the model's learning process.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Scaling numerical columns
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

categories = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'emp_length_to_loan_amnt_ratio', 'int_rate_to_loan_amnt_ratio']

for col in categories:
        df[col] = scalar.fit_transform(df[[col]])
```

<br><br>
### Feature Selection 
**Using chi squared test to assess feature signifiance amongst all the categorical variables**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Performing Chi-Square to see significance with target variable
from sklearn.feature_selection import chi2

x = df[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'age_group', 'income_group']]

y = df['loan_status']

chi_scores = chi2(x, y)

chi_scores
```

<br><br>
**Visualizing the chi value output. In this chart we can see that loan grade and person home ownership have a higher significance**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Higher the Chi-Value, the more significant
chi_values = pd.Series(chi_scores[0], index=x.columns)
chi_values.sort_values(ascending=False, inplace=True)
chi_values.plot.bar()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/55fa2a9826c4aa0cc9c1b5d1b4c4437e92065aa7/Visualizations/Chi%20Value%20Bar%20Chart.png" width="1000" height="1000" />

<br><br>
**Visualizing the p-value output helps determine the statistical significance of features. A p-value below 0.05 indicates that we can reject the null hypothesis, suggesting the feature has a significant relationship with the target variable. In this analysis, the features with the highest chi-squared values also exhibit the lowest p-values, highlighting their strong association with the outcome.**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#P-Value < 0.05 is statistically significant
p_values = pd.Series(chi_scores[1], index=x.columns)
p_values.sort_values(ascending=False, inplace=True)
p_values.plot.bar()
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/55fa2a9826c4aa0cc9c1b5d1b4c4437e92065aa7/Visualizations/P%20Value%20Bar%20Chart.png" width="1000" height="1000" />

<br><br>
**Getting one final understanding of all variable relationships after scaling and encoding**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Visualizing all correlations after scaling and encoding
fig, ax = plt.subplots() 
fig.set_size_inches(15,8)
sns.heatmap(df.corr(), vmax =.8, square = True, annot = True,cmap='Blues', fmt='.2f')
plt.title('Confusion Matrix',fontsize=15);
```

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/55fa2a9826c4aa0cc9c1b5d1b4c4437e92065aa7/Visualizations/All%20Variables%20Confusion%20Matrix.png" width="1350" height="1000" />

<br><br>
## Model Building 
This stage of the project involves importing essential dependencies from scikit-learn for tasks such as model building, model evaluation, and splitting the data into training and testing sets. Although I am still relatively new to the field of machine learning, I selected three widely-used classification algorithms—Logistic Regression, Random Forest Classifier, and XGBoost—to compare their performance. I chose these algorithms because I am working with an imbalanced dataset and wanted to observe the contrast between a simpler algorithm like Logistic Regression and more advanced tree-based/gradient boosting models like Random Forest and XGBoost. The latter two are better suited for handling imbalanced data, thanks to their capabilities such as class weight adjustment and cost-sensitive learning techniques.

<br><br>
**Initializing Logistic Regression and fitting training data**
<div style="max-height: 400px; overflow-y: auto;">
  
```python
#Import scikit-learn libraries for machine learning implementation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Splitting variables into predictors and target
X = df.drop(columns = 'loan_status')
y = df['loan_status']

#Splitting data into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initializing model and fitting training data
model = LogisticRegression()
model.fit(X_train, y_train)

#Using K-Fold to assess based off subsets of training data
k = 5
cv_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)

report = classification_report(y_test, y_predict)

#Outputting evaluation scores
print(f'Cross-Validation accuracy scores: {cv_scores}')
print(f'Average of Cross-Validation accuracy scores: {cv_scores.mean()}')
print(f'Test Set accuracy: {accuracy}')
print(f'Report: {report}')
```

<br><br>
**Initializing Random Forest and fitting training data**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Importing Random Forest algorithm
from sklearn.ensemble import RandomForestClassifier

#Initializing model and fitting training data
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

#Using K-Fold to assess based off subsets of training data
k = 5
cv_scores2 = cross_val_score(model2, X_train, y_train, cv=k, scoring='accuracy')

y_predict2 = model2.predict(X_test)

accuracy2 = accuracy_score(y_test, y_predict2)
report2 = classification_report(y_test, y_predict2)

#Outputting evaluation scores
print(f'Cross-Validation accuracy scores: {cv_scores2}')
print(f'Average of Cross-Validation accuracy scores: {cv_scores2.mean()}')
print(f'Test Set accuracy: {accuracy2}')
print(f'Report: {report2}')
```

<br><br>
**Initializing XGBoost and fitting training data.**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Importing and evaluating XGBoost 
import xgboost as xgb

model3 = xgb.XGBClassifier()

model3.fit(X_train_reduced, y_train)
y_predict3 = model3.predict(X_test_reduced)

accuracy3 = accuracy_score(y_test, y_predict3)
report3 = classification_report(y_test, y_predict3)

print(f'Test Set accuracy: {accuracy3}')
print(f'Report: {report3}')
```

<br><br>
## Model Evaluation 
In this critical stage of the project I conducted a comprehensive evaluation of all the models to identify the best performing one. The techniques I utilized were Accuracy, Classification Report, K-Fold Cross Validation. I also employed Feature Importance as a way of identifying the most critical features so I could reduce parameters while maintaining information. 

<br><br>
**Logistic Regression Results**

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/bea0f0b183a79e79837a9891195aa711b0c850fc/EvaluationMetrics/LogisticRegression%20Eval.png" width="700" height="300" />

<br><br>
**Random Forest Results**

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/bea0f0b183a79e79837a9891195aa711b0c850fc/EvaluationMetrics/Screenshot%202025-02-12%20150959.png" width="700" height="300" />

<br><br>
**Random Forest Results Using Only Top 5 Important Features**

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/1a0732ac7637846c672f906d6e265369bbc09a1c/EvaluationMetrics/Screenshot%202025-02-12%20151142.png" width="700" height="300" />

<br><br>
**XGBoost Results**

<img src="https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/1a0732ac7637846c672f906d6e265369bbc09a1c/EvaluationMetrics/Screenshot%202025-02-12%20151205.png" width="700" height="300" />

<br><br>
## Findings 
In my analysis, Logistic Regression achieved the lowest accuracy at 85%, while Random Forest performed the best with 92% accuracy. XGBoost, using only the most important features, came in slightly lower at 90%. However, when I trained the Random Forest model using only the most important features, I noticed a significant drop in precision for the minority class (Credit Risk), from 95% to 82%. This suggests that while some features may not have high importance ratings, retaining a larger number of parameters could still play a crucial role in accurately identifying loan defaults.

In the context of credit risk assessment, maintaining high precision is critical, as misclassifications could lead to substantial financial losses for the bank. Therefore, it may be more beneficial to retain all features rather than reducing dimensionality, as this helps preserve the model's accuracy and predictive power.

Additionally, the results indicate that Random Forest and XGBoost outperform Logistic Regression on this dataset. This could be attributed to the medium-sized dataset (30,000 rows) or the ability of these algorithms to better handle class imbalances. As I continue to explore different algorithms and business cases, I aim to deepen my understanding of when and why certain methods are more effective, further refining my approach to solving complex problems.

## Future Work and Improvements 
Additional things that I would like to implement in this project are
- Hyperparamater Tuning: Utilizing GridSearchCV to test every combination of model parameters to identify the configuration that yields the highest accuracy. This approach is one of the most efficient ways to evaluate different parameter sets while minimizing manual effort. Incorporating paramater tuning techniques into the machine learning workflow is essential for optimizing model performance, and it’s a technique I’m eager to apply in future projects to ensure the best possible results.

- Model Deployment: Creating a web flask server in Python to integrate my machine learning model and deployed it on a cloud server instance. This would allows users to make API calls to the server and receive predictions from the model. Incorporating this into my next project would be an exciting way to learn more about deploying and managing models in a production environment, giving me valuable hands-on experience with real-world applications.

## Learnings 
Concepts I learned about through this project:
  - Data Cleaning
  - Data Visualization
  - Handling Outliers
  - Feature Engineering techniques such as binning, merging features, scaling, and encoding
  - Dimensionality
  - Feature Selection 
  - Hypothesis Testing
  - Chi Square Test
  - Random Forest / XGBoost
  - Precision, Recall, F-1 Score
  

