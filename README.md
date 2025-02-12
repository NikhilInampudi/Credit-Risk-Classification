# Credit Risk Classification Project
![Credit Risk Picture](https://github.com/NikhilInampudi/Credit-Risk-Classification/blob/50634a931067490d3a5fe12af0637f24ac896de1/Credit%20Risk%20Image.jpg)

## Table of Contents
- [Overview](#overview)
- [Tools Used](#tools-used)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Data Collection and Cleaning](#data-collection-and-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Pre-Processing](#pre-processing)
- [Model-Building](#model-building)
- [Model-Evaluation](#model-evaluation)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Work](#limitations-and-future-work)
- [Conclusion](#conclusion)

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
**Plotly pie chart to visualize distributino of home ownership. This tells us an overwhelming majority of people are stil paying for their living situation. This is often a huge expense for someone to pay for and could have an effect on someones likelihood of defaulting on a loan as they have less leftover income.**

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
**Plotly pie chart to visualize distributino of different loan grades. Based on this results we can see that around 80% of individuals have a satisfactory rating (A-C) and a mminority of individuals have a bad rating (D-G). This would probably have a big effect on credit worthinesss as this is going off of someones prior history of using credit.**

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



