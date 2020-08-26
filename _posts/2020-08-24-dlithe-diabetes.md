---
title: "Binary classification on PIMA Indian Dataset"
date: 2020-08-24
tags: [dlithe internship, data science, decision tree classifier]
header:
  image: "/images/default.jpg"
excerpt: "The goal is to predict whether or not a given female patient will contract diabetes based on certain features."
mathjax: "true"
---

This problem is comprised of 768 observations of medical details for Pima indians patents. The records describe instantaneous measurements taken from the patient such as their age, the number of times pregnant and blood workup. All patients are women aged 21 or older. All attributes are numeric, and their units vary from attribute to attribute.<br><br>
Each record has a class value that indicates whether the patient suffered an onset of diabetes within 5 years of when the measurements were taken (1) or not (0).<br><br>
The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies. Therefore, it is a binary classification problem. A target value of 0 indicates that the patient does not have diabetes, while a value of 1 indicates that the patient does have diabetes.<br><br>
There may be some missing values with which you have to deal with.<br><br>
Build a prediction Algorithm using Decision Tree.<br>


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
```


```python
df=pd.read_csv(r"D:\Learning\DLithe-ML\Assignment\diabetes.csv")
```


```python
df.shape
```




    (768, 9)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
df=df.drop_duplicates()
df.shape
```




    (768, 9)




```python
df.isnull().sum()
```




    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64




```python
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
      <th>Pregnancies</th>
      <td>6.000</td>
      <td>1.000</td>
      <td>8.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>148.000</td>
      <td>85.000</td>
      <td>183.000</td>
      <td>89.000</td>
      <td>137.000</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>72.000</td>
      <td>66.000</td>
      <td>64.000</td>
      <td>66.000</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>35.000</td>
      <td>29.000</td>
      <td>0.000</td>
      <td>23.000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>94.000</td>
      <td>168.000</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>33.600</td>
      <td>26.600</td>
      <td>23.300</td>
      <td>28.100</td>
      <td>43.100</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>0.627</td>
      <td>0.351</td>
      <td>0.672</td>
      <td>0.167</td>
      <td>2.288</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>50.000</td>
      <td>31.000</td>
      <td>32.000</td>
      <td>21.000</td>
      <td>33.000</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print number of missing values by count
print((df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] == 0).sum())
```

    Glucose                       5
    BloodPressure                35
    SkinThickness               227
    Insulin                     374
    BMI                          11
    DiabetesPedigreeFunction      0
    Age                           0
    dtype: int64
    


```python
# Replace all 0 with NaN
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
```


```python
# print the first 5 rows of data
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
      <th>Pregnancies</th>
      <td>6.000</td>
      <td>1.000</td>
      <td>8.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>148.000</td>
      <td>85.000</td>
      <td>183.000</td>
      <td>89.000</td>
      <td>137.000</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>72.000</td>
      <td>66.000</td>
      <td>64.000</td>
      <td>66.000</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>35.000</td>
      <td>29.000</td>
      <td>NaN</td>
      <td>23.000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94.000</td>
      <td>168.000</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>33.600</td>
      <td>26.600</td>
      <td>23.300</td>
      <td>28.100</td>
      <td>43.100</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>0.627</td>
      <td>0.351</td>
      <td>0.672</td>
      <td>0.167</td>
      <td>2.288</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>50.000</td>
      <td>31.000</td>
      <td>32.000</td>
      <td>21.000</td>
      <td>33.000</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# count the number of NaN (null) values in each column
print(df.isnull().sum())
```

    Pregnancies                   0
    Glucose                       5
    BloodPressure                35
    SkinThickness               227
    Insulin                     374
    BMI                          11
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                       0
    dtype: int64
    


```python
# Replace the null values with the mean of that column
df = df.fillna(df.mean())

# Check the number of null values after replacing
print(df.isnull().sum())
```

    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64
    


```python
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
      <th>Pregnancies</th>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>148.000000</td>
      <td>85.000000</td>
      <td>183.000000</td>
      <td>89.000</td>
      <td>137.000</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>72.000000</td>
      <td>66.000000</td>
      <td>64.000000</td>
      <td>66.000</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>35.000000</td>
      <td>29.000000</td>
      <td>29.153420</td>
      <td>23.000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>155.548223</td>
      <td>155.548223</td>
      <td>155.548223</td>
      <td>94.000</td>
      <td>168.000</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>33.600000</td>
      <td>26.600000</td>
      <td>23.300000</td>
      <td>28.100</td>
      <td>43.100</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>0.627000</td>
      <td>0.351000</td>
      <td>0.672000</td>
      <td>0.167</td>
      <td>2.288</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>50.000000</td>
      <td>31.000000</td>
      <td>32.000000</td>
      <td>21.000</td>
      <td>33.000</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
          dtype='object')




```python
a = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
b = ['Outcome']
```


```python
plt.rcParams['figure.figsize'] = (20, 10)
df.hist()
plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_16_0.png)



```python
plt.rcParams['figure.figsize'] = (6, 5)
sns.countplot(x="Outcome",data=df)
plt.show()
print(df["Outcome"].value_counts())
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_17_0.png)


    0    500
    1    268
    Name: Outcome, dtype: int64
    

> This shows that there are 500 negative results and 268 positive results


```python
for i in a:
    sns.swarmplot(x="Outcome",y=i,data=df)
    plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_0.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_1.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_2.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_3.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_4.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_5.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_6.png)



![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_19_7.png)


No clear conclusions can be drawn from the above swarmplots<br>
But these are some readings that could prove useful:
> Women with more than 13 pregnancies tend to get diabetes<br>
Women with glucose less than 75 tend to NOT get diabetes.


```python
for i in a:
    print(df[i].describe())
    print(df[i].skew())
    sns.distplot(df[i], kde=False)
    plt.show()
```

    count    768.000000
    mean       3.845052
    std        3.369578
    min        0.000000
    25%        1.000000
    50%        3.000000
    75%        6.000000
    max       17.000000
    Name: Pregnancies, dtype: float64
    0.9016739791518588
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_1.png)


    count    768.000000
    mean     121.686763
    std       30.435949
    min       44.000000
    25%       99.750000
    50%      117.000000
    75%      140.250000
    max      199.000000
    Name: Glucose, dtype: float64
    0.5327186599872982
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_3.png)


    count    768.000000
    mean      72.405184
    std       12.096346
    min       24.000000
    25%       64.000000
    50%       72.202592
    75%       80.000000
    max      122.000000
    Name: BloodPressure, dtype: float64
    0.13730536744146796
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_5.png)


    count    768.000000
    mean      29.153420
    std        8.790942
    min        7.000000
    25%       25.000000
    50%       29.153420
    75%       32.000000
    max       99.000000
    Name: SkinThickness, dtype: float64
    0.8221731383793047
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_7.png)


    count    768.000000
    mean     155.548223
    std       85.021108
    min       14.000000
    25%      121.500000
    50%      155.548223
    75%      155.548223
    max      846.000000
    Name: Insulin, dtype: float64
    3.019083661355125
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_9.png)


    count    768.000000
    mean      32.457464
    std        6.875151
    min       18.200000
    25%       27.500000
    50%       32.400000
    75%       36.600000
    max       67.100000
    Name: BMI, dtype: float64
    0.5982526551146302
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_11.png)


    count    768.000000
    mean       0.471876
    std        0.331329
    min        0.078000
    25%        0.243750
    50%        0.372500
    75%        0.626250
    max        2.420000
    Name: DiabetesPedigreeFunction, dtype: float64
    1.919911066307204
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_13.png)


    count    768.000000
    mean      33.240885
    std       11.760232
    min       21.000000
    25%       24.000000
    50%       29.000000
    75%       41.000000
    max       81.000000
    Name: Age, dtype: float64
    1.1295967011444805
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_21_15.png)


From the above distplots, we can make the following conclusions:<br>

> Distplot 1 : Pregnancies
>>Least Pregnancy: 0<br>
Highest Pregnancy: 17<br>
Average Pregnancy: 4<br><br>
Since it's positively skewed - Majority of the women have fewer pregnancies.

> Distplot 2 : Glucose
>>Least Glucose level: 44<br>
Highest Glucose level: 199<br>
Average Glucose level: 121.68<br><br>
Since it's positively skewed - Majority of the women have lower glucose level.

> Distplot 3 : Blood Pressure
>>Least Blood Pressure: 24 mm Hg<br>
Highest Blood Pressure: 122 mm Hg<br>
Average Blood Pressure: 72.40 mm Hg<br><br>
Since it's positively skewed - Majority of the women have a lower Blood Pressure.

> Distplot 4 : Skin Thickness
>>Least Skin Thickness: 7 mm<br>
Highest Skin Thickness: 99 mm<br>
Average Skin Thickness: 29.15 mm<br><br>
Since it's positively skewed - Majority of the women have a lower Skin Thickness.

> Distplot 5 : Insulin
>>Least Insulin level: 14 U/ml<br>
Highest Insulin level: 846 U/ml<br>
Average Insulin level: 155.54 U/ml<br><br>
Since it's positively skewed - Majority of the women have a lower insulin level.

> Distplot 6 : BMI
>>Least BMI: 18.20<br>
Highest BMI: 67.10<br>
Average BMI: 32.45<br><br>
Since it's positively skewed - Majority of the women have a lower BMI.

> Distplot 7 : DiabetesPedigreeFunction
>>Least value: 0.078<br>
Highest value: 2.420<br>
Average value: 0.471<br><br>
Since it's positively skewed - Majority of the women have a lower DiabetesPedigreeFunction value.

> Distplot 8 : Age
>>Least age: 21<br>
Highest age: 33<br>
Average age: 81<br><br>
Since it's positively skewed - Majority of the women are younger.


```python
print(df.skew())
```

    Pregnancies                 0.901674
    Glucose                     0.532719
    BloodPressure               0.137305
    SkinThickness               0.822173
    Insulin                     3.019084
    BMI                         0.598253
    DiabetesPedigreeFunction    1.919911
    Age                         1.129597
    Outcome                     0.635017
    dtype: float64
    

>Insulin and DiabetesPedigreeFunction are highly skewed data. So we will normalize by using the log values of the colulmns


```python
df['Insulin'] = np.log(df['Insulin'])
print("skew: ", df['Insulin'].skew())
sns.distplot(df['Insulin'], kde=False)
plt.show()

print('-'*40)

df['DiabetesPedigreeFunction'] = np.log(df['DiabetesPedigreeFunction'])
print("skew: ", df['DiabetesPedigreeFunction'].skew())
sns.distplot(df['DiabetesPedigreeFunction'], kde=False)
plt.show()
```

    skew:  -0.7896950416949489
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_25_1.png)


    ----------------------------------------
    skew:  0.11417768826564408
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_25_3.png)


<h2>Machine Learning model - Decision Tree</h2>


```python
# Split the dependent and independent values
x = df.drop("Outcome",axis=1)
y = df["Outcome"]
```


```python
# pre-processing the data
x = StandardScaler().fit(x).transform(x)
```


```python
# Split the data for training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7)
```


```python
print ('Train set:', xtrain.shape,  ytrain.shape)
print ('Test set:', xtest.shape,  ytest.shape)
```

    Train set: (537, 8) (537,)
    Test set: (231, 8) (231,)
    


```python
# Load decision tree classifier model from sklearn and fit the training sets
algo = DecisionTreeClassifier(criterion='entropy',max_depth=5)
algo.fit(xtrain,ytrain)
```




    DecisionTreeClassifier(criterion='entropy', max_depth=5)




```python
plt.figure(figsize=(20,10))
plot_tree(algo,filled=True)
plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-diabetes/DLithe-ASSIGNMENT-diabetes_32_0.png)



```python
ypred = algo.predict(xtest)

# compare predicted values and actual values and find out accuracy
print("Accuracy: ", accuracy_score(ytest,ypred))
```

    Accuracy:  0.7445887445887446
    


```python
print(confusion_matrix(ytest,ypred))
```

    [[112  38]
     [ 21  60]]
    

### Highest Accuracy:  0.7662337662337663