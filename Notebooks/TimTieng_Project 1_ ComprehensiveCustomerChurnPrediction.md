# Project 1: Comprehensive Customer Churn Prediction

## Project Overview

In this project activity, you will dive into analyzing customer data from a business to predict churn risk comprehensively. You will not only apply traditional supervised learning algorithms such as decision trees and random forests but also explore logistic regression, support vector machines (SVMs), and k-nearest neighbors (KNN) to identify key churn factors. This expanded scope will enhance your understanding of different model assumptions, strengths, and weaknesses, enabling you to build a robust model to target potential churners effectively.

In this project, you will:
- **Evaluate** different supervised learning algorithms to understand their suitability for churn prediction.
- **Implement** data preprocessing techniques tailored to the requirements of each algorithm.
- **Optimize** model parameters to improve prediction accuracy.
- **Interpret** the model outcomes to extract actionable insights for retention campaigns.

**Estimated Completion Time**

12 to 14 hours


### Project Information

    Python Version: 3.10.13
    Python Packages: Outlined in pip_requirements.txt
    Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv (Provided)
    File Authors: Tim Tieng, Scott Mayer


```python
# Import Packages: 
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Algorithms, Modeling and preprocessing packages
import feature_engine
from scipy.stats import anderson
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Deep Learning
import keras
```


## Task 1: Initial Data Exploration

**Objective:** Load the dataset and perform an initial examination to understand its structure and identify any immediate cleaning needs.

**Activities:**

1. Load the dataset using pandas and display the first few rows to get an initial understanding of the data.
2. Examine the dataset's shape to understand the scale of the data we're dealing with.
3. Check the data types of each column to identify which are categorical and which are numerical.
4. Identify any missing values in the dataset.
5. Generate summary statistics for numerical columns to identify any immediate anomalies or outliers.


**Estimated Completion Time:** 60 minutes

**Hints:**

* Use `pd.read_csv()` to load the dataset. Remember to `import pandas as pd`.
* Use `.head()`, `.info()`, `.dtypes`, `.isnull().mean()`, and .`describe()` methods to explore the dataset.

### Task 1.1: Load the Dataset


```python
# Obtain - Read in the data, convert to PD Dataframe, and perform initial inspection of the dataset
churn_file = pd.read_csv('../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn_df = pd.DataFrame(churn_file)

```

### Task 1.2: Examine the Dataset:


```python
print(churn_df.shape)
```

#### Observations
1. Rows/Observations: 7043 
2. Attributes/Columns: 21
3. Datatypes Present: int64(2x), float64(1x), string objects(18x)
4. Memory Usage - 7.8 MB (Small)


### Task 1.3: Check Datatypes


```python
# Initial inspect - Provde information on Attribute Names, Non-Null Count, Data Types, Memory Usage
churn_df.info(memory_usage='deep')

```

### Task 1.4: Check for Missing Values

#### Observations 
TotalCharges has some values with whitespace strings. These need replaced with 0s. We will also change the SeniorCitizen Column to a string because although it contains 0s and 1s, it is a categorical column. Lastly, we are changing TotalCharges to a numerical column because it contains numerical data. 


```python
# Check for null Values - percentage per column
churn_df.isna().mean().sort_values(ascending= False)
# No null values 

```


```python

churn_df.TotalCharges = churn_df['TotalCharges'].replace(' ', 0)
churn_df['SeniorCitizen'] = churn_df['SeniorCitizen'].astype('str')
churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'])
churn_df.info(memory_usage='deep')
```


### Task 1.5: Generate Summary Statistics


```python
# Intial Inspection of data using head()
churn_df.head()
```


```python

# Desriptive Statistics -Numerical Attributes
churn_df.describe()
```


```python
# Descriptive Statistics - Categorical Values included
churn_df.describe(include='all')
```

#### Data Observerations

1. Gender Distribution - Dataset has 7043 total observations with only two unique values (Male, Female). Right now there are 3555 Male values Which suggest the data is relatively balanced. 
2. Senior Citizen Attribute (Prior to DT casting) - The mean value is .162 which can be translated as 16.2 of the userbase is classified as a senior citizen. Additional definition of what describes a Senior Citizen mayh be needed for future analysis (what age classifies senior citizen cut off?)
3. Tenure - Mean value is about 32.4% with a min value of 0 months and max months membership of 72 months. STD is 24.6 which can be viewed as a wide variance in the quartiles.
4. XService Attributes - This showcases that this one company has multiple offerings  that can appeal to different customer bases. From a business perspective, we may need to study customer propensity to include one service based on the services the customer currently pays for. This can increase market capture for the business writ-large. (Bundling services to a subscription)
5. InternetService - Fiber Optic is the most frequent value in this column representing that almost half of the customer base pays for fiber optic services. This could be a business opportunity to focus on members with lower levels of internet service to upgrade potentially reducing the likliness of leaving our service for a competitor.
5. Churn - Top value is "no" with a frequency count of 5174/7043. This attribute should be used in future analysis

---
---


## Task 2: Exploratory Data Analysis (EDA)


**Objective:** Use statistical analysis and visualization techniques to uncover insights and identify patterns related to custom	er churn.

**Activities:**

1. **Visualize distribution of numerical features** to identify any skewed data or outliers.
2. **Analyze churn rate by categorical features** to uncover any patterns that may indicate a higher likelihood of churn.
3. **Examine relationships between features** using correlation matrices and scatter plots for numerical features, and stacked bar charts for categorical features against churn.
4. **Use box plots to identify outliers** in numerical data and understand distributions across churned and retained customers.
5. **Create a pair plot** to visualize the pairwise relationships and distributions of numerical features segmented by churn.


**Estimated Completion Time:** 90 minutes


**Hints:**

* Use `sns.histplot()` for distributions, `sns.countplot()` for categorical data analysis, `sns.heatmap()` for correlation matrices, `sns.boxplot()` for outliers, and `sns.pairplot()` for pairwise relationships.
* Remember to import necessary libraries like `seaborn as sns` and `matplotlib.pyplot as plt`.



### Task 2.1 Visualize distribution of numerical features
Created a function to visualize numerical features. 



```python

def visualize_numerical_histograms(df):
    """
    Purpose - A function that takes in a dataframe and returns histograms on the dataframe's numerical values. 
              This helps with Exploratory Data Analysis (EDA).
    Parameters - Pandas Dataframe
    Returns - Nothing
    Prints - Histograms to highlight distributions
    """
    # Create numerical only dataframe based on datatypes in the info() output
    numerical_only_df = churn_df.select_dtypes(include=['float64', 'int64'])

    # Iterate
    for column in numerical_only_df.columns:
        # format the visualization
        plt.figure(figsize=(10,6))
        sns.histplot(data=numerical_only_df, x=column, kde=True, bins= 20)
        plt.title(f"Distribution of {column} Column of Churn Dataset")
        plt.xlabel = column
        plt.ylabel = 'Frequency'
        plt.show()

```


```python
# Test and Verifify
visualize_numerical_histograms(churn_df)
```

#### Visual Observations

1. **Tenure Distribution**: There are two distinct peaks for this attribute, alluding to that the current customer base are either new OR customers that churn quickly. The center dip in the graph may allude to the typical range where churn is likely to happen
2. **MonthlyCharges Distribution**: Visually, there is a right skew to this column. There also seems to be a lot of customers who are only paying for $20 for services. This may highlight that these customers are only paying for a single service (internet, phone, etc). From a business/market capture perspective, we can view this as potential customer base to target with the goal of motivating them to add more services to their subscription. The right end of the graph may represent customers who pay for premium-like services or combines multiple service offerings to their subscription. 
3. **TotalCharges Distributio**n**: This attribute has a long trailing tail towards the right of the graph. Additionally, there are alot of customers with a low "TotalCharges" value. This is aligned with teh MonthlyCharges column and may be attributed to the customer base only paying for 1 service. 

**Business Significance**: The three graphs shows information abou the current customer base. It seems as though there are two "populations" in the customer base: Customers who are new, and customers who pay for multiple services. Based on the dip in the MonthlyCharges Distribution, this could be viewed as our customberbase we should focus on.


#### Statistical Observations: Anderson Tests for Skewness



```python
def anderson_skewness_test(df):
  """
  Purpose - Conducts anderson skewness test on numerical columns to determine if column is normally distrubuted
  Parameters - Pandas Dataframe
  Returns - Nothing
  Prints - column name, anderson value, whether or not column is normal 
  """
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  for col in df.select_dtypes(include='number'):
    a = anderson(df[col], dist='norm')
    print(col + "\t", end="")
    if len(col) < 8:
      print("\t", end="")
    print(f'{a.statistic:.2f}', end="\t")

    if a.statistic < a.critical_values[0]:
      print("Normal")
    else:
      print("Not Normal")
  return 

anderson_skewness_test(churn_df)
```


Observations: 
- None of our numerical columns are normal at the 15% significance level based on having anderson values above the critical value at the 15% threshold. 

### Task 2.2 Analyze Churn by Categorical Features


```python
def analyze_churn_categorical_features(df):
  """
  Purpose - To identify distribution of categories in categorical features
  Parameters - Pandas Dataframe
  Returns - Nothing
  Prints - Boxplots for the provided pandas Dataframe that can be used for Exploratory Data Analysis (EDA)
  """
  categorical_cols = []
  numerical_cols = []
  for c in df.columns:
    if df[c].map(type).eq(str).any() or df[c].map(type).eq(str).any():
      categorical_cols.append(c)
    else:
      numerical_cols.append(c)

  for col in categorical_cols:
    count = df[col].nunique() 
    print(f"{col}", end="")
    if len(col) < 8:
      print("\t", end="")
    if len(col) < 16:
      print("\t", end="")
    print(f" Num Categories: {count}")
    for cat in df[col].unique():
      if col == "customerID":
        continue
      count_cat = df[col].value_counts()[cat]  
      print(f"\t{cat}: {count_cat}, {count_cat/df.shape[0]*100:.2f}%")
    print()
    
  data_numeric = df[numerical_cols]
  data_categorical = pd.DataFrame(df[categorical_cols])
    
  data_joined = pd.concat([data_numeric, data_categorical], axis=1)
    
  data_joined.describe(include='all')
  data_joined.head()
  return

```


```python
analyze_churn_categorical_features(churn_df)
```

#### Observations 
- **Encoding Techniques**: Based on the distributions, it is likely a LabelEncoder is the best way to encode the various categorial columns in the dataset. 

### Task 2.3: Use Boxplots to identify outliers in numerical data


```python
# Visualize Box Plots
def create_boxplots(df):
    """
    Purpose - To create boxplots of a given dataframe
    Parameters - Pandas Dataframe
    Returns - Nothing
    Prints - Boxplots for the provided pandas Dataframe that can be used for Exploratory Data Analysis (EDA)
    """
    # Create numerical only dataframe
    numerical_only_df = churn_df.select_dtypes(include=['float64', 'int64'])
    # Base figure width that can be  scale based on the range of the data
    base_width = 5
    max_width = 15 # added to resolve ValueError: Image size of 217119x400 pixels is too large. It must be less than 2^16 in each direction.

    # Iterate over each column
    for column in numerical_only_df:
        # Calculate the range of the column data
        data_range = numerical_only_df[column].max() - numerical_only_df[column].min()
        
        # Scale the figure width, we use max to ensure a minimum width is maintained
        figure_width = min(max(base_width, base_width * (data_range / 20)), max_width)  # Updated as part of max_width 
        
        # Create a figure with the adjusted size
        plt.figure(figsize=(figure_width, 4))  # Height is kept constant
        plt.title(f'Boxplot for {column} Attribute')
        sns.boxplot(x=df[column], orient='h')
        
        plt.show() 
```


```python
create_boxplots(churn_df)
```


#### Boxplot Observations

1. **SeniorCitizen (Prior to casting)** - There seems to be an outlier that returns an ineffective boxplot. I need to create a function that can handle outliers. I originally thought handling the figure width was needed to resolve the error. 
2. **Tenure Boxplot** : This attrribute looks relatively uniform. This graph visually describes the customers who been wiht the services for a long time
3. **MonthlyCharges Boxplot**: The median placement on the graph confirms the right skew nature from the histogram. This plot also confirms the observation that customers tend to have lower monthly charges or are paying for single services vs bundling services
4. **TotalCharges Boxplot**: The skewness of this attribute is more apparent in the boxplot. The median value represents that half of the customer base is clustered towards the lower end of charges. Potential outliers as indicated by the long tail on the right side of the plot


```python
# Correlation matrix and heatmap visualization
correlation_matrix = churn_df.corr(numeric_only=True)
correlation_matrix
```


```python
# Heatmap visualization of the correlation matrix 
# Purpose of visualizing Correlation Matrix - To see the relationships more apparently 
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

### Correlation Matrix and Heatmap Observations:
1. There are positive relationships amongst the 3 numerical attributes.
2. **SeniorCitizen vs Tenure (Prior to Casting)e** = The value is so close to 0 that it alludes to there is no relationship at all between these two attributes 
3. Tenure vs Monthly Charges - There is a slightly positive relationship between these two attributes. This suggests that as tenure increases, monthyl charges also increase slightly since they are positively correlated, although weak. This is an interesting anamoly as businesses tend to "reward" long-term customers or value their business throughout the years. This could be the business' approach to handle inflation.


```python
def chi_squared_test(df, target):
    """
    Purpose - to assist in EDA on categorical data. This function performs a Chi-squared test on all all categorical 
            attributes in a df against the target variable.
    Target Variable - 'Churn'
    Parameters- pandas dataframe with numerical and categorical data
    Return - results of the chi2 test in the 'results' variable
    Prints - The Chi-squared statistic and p-value for each pair of categorical columns.
    """
    # Select Object dt based off the output from info()
    categorical_cols = df.select_dtypes(include=['object']).columns
    significant_results = []

    # Check if 'Churn' Column is present. If not, prompt the user
    if target not in categorical_cols:
        print(f"The target variable {target} is not found. Please verify")
        return significant_results

    # Perform a Chi-squared test for each pair of categorical variables
    for col in categorical_cols:
        if col == target:
            continue
        # Create a contingency table
        contingency_table = pd.crosstab(df[col], df[target])

        # Perform the Chi-squared test - chi2 value, p value, degrees of freedom and expected frequencies if null hypothesis were true
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Save the result IF p value is less than or equal to 0.05 - Statistically significant
        if p <= 0.05:
            significant_results.append({
                'Column 1': col,
                'Chi-squared Statistic': chi2,
                'p-value': p
            })

            # Output the result
            print(f"Chi-squared test for {col} and {target}")
            print(f"Chi-squared Statistic: {round(chi2,2)}, p-value: {round(p,4)}\n")

    return significant_results
```


```python
# Test by calling the function
chi_squared_test(churn_df, 'Churn')
```

### Chi-Squared On Categorical Values Observations

- To identify the relationship between categorical values and our target variable of "Churn", we created a function to run the chi-squared test on our dataframe.
- Low p-values across the attributes, confirming strong rejection of the null hypothesis
- These attributes may have strong predicitve power in our future modeling

**Attributes with strong Association with 'Churn' (Chi Statistic values are 500 or greater)**
1.  InternetService
2.  OnlineSecurity
3.  OnlineBackup
4.  DeviceProtection
5.  TechSupport
6.  PaymentMethod
7.  Contract


### Cramer's V-Test on Categorical Values


```python
# Cramers-V Test Function Version 2
def cramers_v(confusion_matrix):
    """
    Purpose: Calculates Cramer's V statistic for categorical variables.
    Parameters: 
        x = first categorical Value
        y = second categorical value
    Returns: Cramer's V value
    """
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()  # Total observations
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

### Task 2.5: Create Pair Plot


```python
def create_pair_plot(df): 
    """
    Purpose - 
    Parameters - 
    Return - 
    Prints - 
    """
    pass

```


```python

def calculate_cramers_v_for_attributes(df, attributes):
    """
    Purpose: To assess the relationship between each pair of categorical attributes.
    Parameters:
        df: pandas DataFrame containing the data.
        attributes: list of column names of potential categorical variables.
    Returns:
        A dictionary with key as attribute pairs and value as Cramer's V statistic.
    """
    cramers_v_results = {}
    for i, attr1 in enumerate(attributes):
        for attr2 in attributes[i+1:]:
            confusion_matrix = pd.crosstab(df[attr1], df[attr2])
            cramers_v_value = cramers_v(confusion_matrix)
            cramers_v_results[(attr1, attr2)] = cramers_v_value

    return cramers_v_results
```


```python
# Test
potential_attributes = ['InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'PaymentMethod', 'Contract' ]
cramers_v_results = calculate_cramers_v_for_attributes(churn_df, potential_attributes)
for pair, value in cramers_v_results.items():
    print(f"Cramer's V for {pair}: {value:.4f}")
```

### Cramer's V Interpretation

Notes - Cramer's V is used to assess the strength of association between pairs of categorical values in our churn_df. Cramer's V helps identify potentially redundant features
If we chose two independent variables that have strong associations to each other, the model may bias towards those attributes, affecting our predicitve power.

V-Values range from approximately 0.2 - 0.7.


```python

```

---
---

### Task 3: Data Preprocessing

create_pair_plot(churn_df)
```

#### Observations

---
---

## Task 3: Data Preprocessing


**Objective:** Split the dataset into training and testing sets, then clean the training dataset by handling missing values, outliers, and duplicate entries using `feature_engine` to prepare for further analysis.

**Activities:**

1. **Split the dataset into training and testing** sets to ensure a fair evaluation of the model built on processed data.
2. **Handle missing values** in the training set using appropriate imputation techniques.
3. **Identify and treat outliers** in numerical features of the training set to minimize their impact.
4. **Check for and remove constant and duplicate entries** in the training set to maintain data integrity.
5. **Apply the same preprocessing steps** (imputation, outlier handling) to the test set using parameters derived from the training set to maintain consistency and prevent data leakage.

**Estimated Completion Time:** 90 minutes

***Note:*** *The splitting of the dataset into training and testing sets before preprocessing is a best practice in machine learning. This approach ensures that the model is evaluated on unseen data that has been processed in the same way as the training data, without using information from the test set during the training phase.*



**Hints:**

* Use `train_test_split` from `sklearn.model_selection` to split your data.
* Handle numerical missing values with pandas `.fillna()`, sklearn's `SimpleImputer` or feature-engine's `MeanMedianImputer`.
* Detect and handle outliers by calculating IQR or using feature_engine's Winsorizer.
* Remove duplicates using DataFrame's `.drop_duplicates()` method or feature-engine's `DropDuplicateFeatures()` class.


```python

# Create a copy of the original data fram for break glass scenarios

churn_original = churn_df.copy()

# Isolate target variable to its own variable with all column values
target = churn_df['Churn']

# Drop Columns

# churn_dropped = churn_df.drop(columns=['Churn', 'customerID', 'gender', 'PaperlessBilling' ,'PaymentMethod'], axis=1, inplace= True)

# Impute if Necessary

# Normalize the Data

# Fit and Transform

# Split the Data


```

---
---

## Task 4: Feature Selection and Engineering

**Objective:** Create new features that might improve model performance, transform applicable features, and select the most relevant features for modeling.

**Activities:**

  1. **Feature Creation:** Create a feature that captures the customer's total spend relative to their tenure. This could highlight customers who might be paying more over a shorter period, potentially indicating dissatisfaction..
  2. **Feature Transformation:**  Normalize skewed features such as MonthlyCharges and TotalCharges using a variance stabilizer to make their distribution more symmetric, which can help certain algorithms perform better.
  3. **Feature Selection:** Use mutual information or another model-based feature selection method to identify features that have the most significant relationship with the target variable, `Churn`.

**Estimated Completion Time:** 90 minutes


```python

```

**Hints:**

* Create new features based on existing data that might indicate behavioral patterns.
* Use feature_engine's `YeoJohnsonTransformer` or manual transformations for skewed features.
* Select relevant features based on mutual information using `SelectKBest` from sklearn.

---
---

## Task 5: Logistic Regression and Assumptions Validation

**Objective:** Implement a logistic regression model and validate its assumptions, adjusting features as necessary.

**Activities:**

  1. **Fit a logistic regression model** using the selected features from Task 4 to predict customer churn.
  2. **Validate that the relationship** between log odds and each independent variable is linear.
  3. **Check for multicollinearity** among predictors.
  4. **Ensure that the residuals (errors) are independent** of each other.
  5. **Check that the variance of error terms is consistent** across all levels of the independent variables.
  6. **Identify and address extreme outliers** that could unduly influence the model.
  7. **Adjust features** based on the findings from assumption validations. This may involve transforming variables, removing or adding features, or addressing outliers and multicollinearity.
  8. **Fit a new logistic regression model** and compare its results with the initial model

**Estimated Completion Time:** 90 minutes


```python

```

**Hints:**

* Fit the logistic regression model using `LogisticRegression` from sklearn.
* Validate linearity using scatter plots or seaborn’s `lmplot`.
* Check multicollinearity with VIF from the statsmodels library.
* Assess model residuals with `residplot` from seaborn or manually plot residuals.

---
---

## Task 6: Decision Trees, Random Forests, and Model Complexity

**Objective:** Build decision tree and random forest models, focusing on understanding and tuning model complexity to avoid overfitting.

**Activities:**

  1. **Use the decision tree classifier** to create a model for predicting customer churn. Focus on understanding the default model complexity and its impact on performance.
  2. **Experiment with parameters** that control the complexity of the decision tree, such as max_depth, min_samples_split, and min_samples_leaf, to find a balance that reduces overfitting while maintaining good predictive performance.
  3. **Implement a random forest classifier** to improve prediction accuracy and robustness by aggregating multiple decision trees.
  4. **Adjust parameters** such as n_estimators, max_depth, and max_features to optimize the random forest model. Aim to enhance model accuracy without significant overfitting.
  5. **Use metrics** such as accuracy, precision, recall, F1 score, and the ROC-AUC score to evaluate and compare the decision tree and random forest models.
  6. **Investigate the features that are most influential** in predicting customer churn according to the random forest model.

**Estimated Completion Time:** 90 minutes



```python

```

**Hints:**

* Train models using `DecisionTreeClassifier` and `RandomForestClassifier` from sklearn.
* Use `GridSearchCV` or `RandomSearchCV` for hyperparameter tuning.
* Evaluate model performance with `sklearn.metrics` and visualize feature importance.

---
---

## Task 7: SVM and KNN Implementation

**Objective:** Apply SVM and KNN algorithms to the churn prediction problem, highlighting the importance of data scaling and parameter tuning.

**Activities:**

  1. **Scale the feature set** to ensure that SVM and KNN algorithms perform optimally, as both are sensitive to the scale of input data.
  2. **Train an SVM model** on the scaled feature set. Start with the default hyperparameters to establish a baseline performance.
  3. **Optimize SVM parameters** including C (regularization parameter) and kernel to improve model performance.
  4. **Apply the KNN algorithm**, initially using a small k (e.g., 5) to model the churn prediction problem.
  5. **Find the optimal k value** for KNN. Consider the balance between underfitting and overfitting as k changes.
  6. **Compare the performance of SVM and KNN models** based on accuracy, precision, recall, F1 score, and ROC-AUC score. Discuss the strengths and weaknesses of each model in the context of the churn prediction problem.
  7. For SVM models, especially linear kernel SVM, **examine the coefficients of features** to understand their impact on the prediction. Use this insight to refine the feature set and improve model simplicity and performance.

**Estimated Completion Time:** 120 minutes


```python

```

**Hints:**

* Scale features using `StandardScaler` or `RobustScaler` before applying `SVM` or `KNN`.
* Train `SVM` using `SVC` and tune with `GridSearchCV` or with `RamdomSearchCV`.
* Implement KNN using `KNeighborsClassifier` and find the best k through cross-validation.
* Compare model metrics using sklearn's evaluation functions.

---
---

##Task 8: Model Evaluation and Comparison

**Objective:** Evaluate the performance of each model using metrics like accuracy, precision, recall, and F1 score, and select the best model based on these metrics.

**Activities:**

  1. **Gather predictions** from all previously implemented models (Logistic Regression, Decision Trees, Random Forests, SVM, KNN) on the test dataset.
  2. **Calculate and compare** the accuracy of each model.
  3. **Compute precision, recall, and F1 scores** for each model to evaluate their performance beyond mere accuracy.
  4. **Calculate the ROC-AUC score** for each model to assess their overall ability to discriminate between positive and negative classes.
  5. **Compare all models** based on the calculated test metrics and select the best performing model(s) for the churn prediction problem.
  6. **Perform an error analysis** on the selected model to identify patterns in the misclassifications.
  7. Based on the evaluation and comparison, **recommend the best model** for predicting customer churn and justify your recommendation.

**Estimated Completion Time:** 90 minutes


```python

```

**Hints:**

* Compile predictions from all models and use `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `roc_auc_score` for evaluation.
* Create a summary table or visualization to compare model performances.
* Conduct error analysis by reviewing the confusion matrix and misclassified examples.

---
---
---
