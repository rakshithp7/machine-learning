---
title: "Classification on Breast Cancer Wisonsin dataset"
date: 2020-08-25
tags: [dlithe internship, data science, logistic regressiona]
header:
  image: "/images/default.jpg"
excerpt: "Breast Cancer Wisonsin data set from UCI machine learning Repository and build a classification model."
mathjax: "true"
---

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
```


```python
df=pd.read_csv(r"D:\Learning\DLithe-ML\Assignment\breast-cancer-wisconsin.csv")
```


```python
df.shape
```




    (699, 11)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 699 entries, 0 to 698
    Data columns (total 11 columns):
     #   Column                       Non-Null Count  Dtype 
    ---  ------                       --------------  ----- 
     0   Sample code number           699 non-null    int64 
     1   Clump Thickness              699 non-null    int64 
     2   Uniformity of Cell Size      699 non-null    int64 
     3   Uniformity of Cell Shape     699 non-null    int64 
     4   Marginal Adhesion            699 non-null    int64 
     5   Single Epithelial Cell Size  699 non-null    int64 
     6   Bare Nuclei                  699 non-null    object
     7   Bland Chromatin              699 non-null    int64 
     8   Normal Nucleoli              699 non-null    int64 
     9   Mitoses                      699 non-null    int64 
     10  Class                        699 non-null    int64 
    dtypes: int64(10), object(1)
    memory usage: 60.2+ KB
    

> "Bare Nuclei" comlumn is of <strong>object</strong> Dtype. Which means it has some invalid(null) values


```python
df=df.drop_duplicates()
df.shape
```




    (691, 11)



> There were 8 duplicate rows


```python
df.isnull().sum()
```




    Sample code number             0
    Clump Thickness                0
    Uniformity of Cell Size        0
    Uniformity of Cell Shape       0
    Marginal Adhesion              0
    Single Epithelial Cell Size    0
    Bare Nuclei                    0
    Bland Chromatin                0
    Normal Nucleoli                0
    Mitoses                        0
    Class                          0
    dtype: int64




```python
df = df.drop("Sample code number", axis=1)
df.head().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Clump Thickness</th>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Uniformity of Cell Size</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Uniformity of Cell Shape</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Marginal Adhesion</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Single Epithelial Cell Size</th>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Bare Nuclei</th>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bland Chromatin</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Normal Nucleoli</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Mitoses</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



> Drop "Sample code number" column since it's not useful in prediction


```python
# Replace all '?' with NaN
df = df.replace({'?':np.nan})

# print the count of null values
print("Null values: ", df["Bare Nuclei"].isnull().sum() ,"\n")

# Convert object dtype to Int64 so we can perform describe() and find out the mean value
df["Bare Nuclei"] = df["Bare Nuclei"].astype(float).astype('Int64')
print(df["Bare Nuclei"].describe())

# Replace the null values with the mean value
print("\nReplacing null values with integer value of mean: ", int(df["Bare Nuclei"].mean()))
df["Bare Nuclei"] = df["Bare Nuclei"].fillna(int(df["Bare Nuclei"].mean()))
```

    Null values:  16 
    
    count    675.000000
    mean       3.537778
    std        3.637871
    min        1.000000
    25%        1.000000
    50%        1.000000
    75%        6.000000
    max       10.000000
    Name: Bare Nuclei, dtype: float64
    
    Replacing null values with integer value of mean:  3
    

<h2>Machine Learning Model - Logistic Regression</h2>


```python
# Split the dependent and independent values
x = df.drop("Class", axis=1)
y = df["Class"]
```


```python
# pre-processing the data
x = StandardScaler().fit(x).transform(x)
```


```python
# Split the data for training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8)
```


```python
print ('Train set:', xtrain.shape,  ytrain.shape)
print ('Test set:', xtest.shape,  ytest.shape)
```

    Train set: (552, 9) (552,)
    Test set: (139, 9) (139,)
    


```python
# Load logistic regression model from sklearn and fit the training sets
algo = LogisticRegression().fit(xtrain,ytrain)
```


```python
# find out the predictions for the testing set
ypred = algo.predict(xtest)

# compare predicted values and actual values and find out accuracy

print("Mean Absolute Error: ", mean_absolute_error(ytest,ypred))
print("Accuracy: ", accuracy_score(ytest,ypred))
```

    Mean Absolute Error:  0.05755395683453238
    Accuracy:  0.9712230215827338
    

# Highest Accuracy:  0.9784172661870504


```python

```
