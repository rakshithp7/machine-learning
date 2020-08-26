---
title: "Binary classification on Titanic Dataset"
date: 2020-08-26
tags: [dlithe internship, data science, logistic regression]
header:
  image: "/images/default.jpg"
excerpt: "The goal is to predict whether a passenger will survive or not using a binary classification model."
mathjax: "true"
---

This is a classic dataset used in many data mining tutorials and demos -- perfect for getting started with exploratory analysis and building binary classification models to predict survival.

*Data covers passengers only, not crew.*


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
```


```python
df=pd.read_csv(r"D:\Learning\DLithe-ML\Assignment\titanic.csv")
```


```python
df.shape
```




    (891, 15)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   survived     891 non-null    int64  
     1   pclass       891 non-null    int64  
     2   sex          891 non-null    object 
     3   age          714 non-null    float64
     4   sibsp        891 non-null    int64  
     5   parch        891 non-null    int64  
     6   fare         891 non-null    float64
     7   embarked     889 non-null    object 
     8   class        891 non-null    object 
     9   who          891 non-null    object 
     10  adult_male   891 non-null    bool   
     11  deck         203 non-null    object 
     12  embark_town  889 non-null    object 
     13  alive        891 non-null    object 
     14  alone        891 non-null    bool   
    dtypes: bool(2), float64(2), int64(4), object(7)
    memory usage: 92.4+ KB
    

<h2>Data cleaning</h2>


```python
# Drop duplicate values

df=df.drop_duplicates()
df.shape
```




    (784, 15)




```python
df.isnull().sum()
```




    survived         0
    pclass           0
    sex              0
    age            106
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    deck           582
    embark_town      2
    alive            0
    alone            0
    dtype: int64



'deck' is mostly empty so we will drop it in the next step.<br>

> Age has 106 null values<br>
embarked and embark_town has 2 null values




```python
# Drop columns which are not needed for our analysis or which are duplicates (another column with values meaning the same)

df=df.drop(["deck", "embarked", "adult_male", "alive", "class"], axis=1)
```

> 'deck' consists of crew data which is excluded in this dataset.

> 'embarked' consists of abbreviated values of 'embark_town'

> 'adult_male' can be analysed from 'who' column; since only a man has adult_male value as 1

> 'alive' is a duplicate of 'survived'

> 'class' is the textual representation of 'pclass'


```python
df.head(10)
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>who</th>
      <th>embark_town</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>man</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>woman</td>
      <td>Cherbourg</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>man</td>
      <td>Southampton</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>man</td>
      <td>Queenstown</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>51.8625</td>
      <td>man</td>
      <td>Southampton</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>child</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>11.1333</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>30.0708</td>
      <td>child</td>
      <td>Cherbourg</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["age"].describe()
```




    count    678.000000
    mean      29.869351
    std       14.759076
    min        0.420000
    25%       20.000000
    50%       28.250000
    75%       39.000000
    max       80.000000
    Name: age, dtype: float64




```python
# Fill the empty values in 'age' column to the mean value of age

print('Replacing null values with mean value : ', int(df["age"].mean()))
df["age"] = df["age"].fillna(int(df["age"].mean()))
df.head(10)
```

    Replacing null values with mean value :  29
    




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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>who</th>
      <th>embark_town</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>man</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>woman</td>
      <td>Cherbourg</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>man</td>
      <td>Southampton</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>man</td>
      <td>Queenstown</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>51.8625</td>
      <td>man</td>
      <td>Southampton</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>child</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>11.1333</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>30.0708</td>
      <td>child</td>
      <td>Cherbourg</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop rows with empty (null) values

df=df.dropna()
```

<h2>Exploring the Data</h2>


```python
a = ['survived', 'pclass', 'sex','sibsp', 'parch', 'who', 'embark_town', 'alone']
b = ['age', 'fare']
```


```python
for i in a:
    sns.countplot(x=i, data=df)
    plt.show()
    print(df[i].value_counts())
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_0.png)


    0    461
    1    321
    Name: survived, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_2.png)


    3    405
    1    212
    2    165
    Name: pclass, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_4.png)


    male      491
    female    291
    Name: sex, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_6.png)


    0    515
    1    201
    2     27
    4     18
    3     14
    5      5
    8      2
    Name: sibsp, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_8.png)


    0    578
    1    114
    2     75
    5      5
    3      5
    4      4
    6      1
    Name: parch, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_10.png)


    man      451
    woman    249
    child     82
    Name: who, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_12.png)


    Southampton    568
    Cherbourg      155
    Queenstown      59
    Name: embark_town, dtype: int64
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_17_14.png)


    True     444
    False    338
    Name: alone, dtype: int64
    

From the above countplots, we can make the following conclusions:

>Survived
>> 461 passengers survived while 321 didnt.

>Passenger Class
>> Majority of the passengers were in 3rd class, followed by 1st class and then 2nd class

> Sex
>> Number of male passengers was more than female

> Port of Embarkation
>> Majority of the passengers embarked from Southampton, followed by Cherbourg and the Queenstown

> Accompanied
>> There were more passengers who travelled alone than with others


```python
for i in b:
    print(df[i].describe())
    print(df[i].skew())
    sns.distplot(df[i], kde=False)
    plt.show()
```

    count    782.000000
    mean      29.700026
    std       13.692729
    min        0.420000
    25%       22.000000
    50%       29.000000
    75%       36.000000
    max       80.000000
    Name: age, dtype: float64
    0.4190340853087404
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_19_1.png)


    count    782.000000
    mean      34.595913
    std       52.176458
    min        0.000000
    25%        8.050000
    50%       15.875000
    75%       33.375000
    max      512.329200
    Name: fare, dtype: float64
    4.583205969233933
    


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_19_3.png)


From the above countplots, we can make the following conclusions:
> Age
>> Youngest: 0.42 (5 months) <br>
Average age: 29<br>
Oldest: 80

> Fare
>> Lowest: 0 (free)<br>
Average: 34.5<br>
Highest: 512<br><br>
The highly positive skewed data (and the info shown above) proves that 75% of the passengers paid a fare lesser than 34. 


```python
sns.swarmplot(x="survived", y="fare", data=df)
plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_21_0.png)


> From the figure, we can come to a conclusion that there were a few set of 'elite' passengers who paid a very high fare, and were able to survive.


```python
df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean()
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
      <th>pclass</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.627358</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.509091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.256790</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(x='pclass',hue="survived", data=df)
plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_24_0.png)


> Survival rate was highest in 1st (upper) class and lowest in 3rd class.
>>Highest number of passengers who survived were from 1st class.<br><br>
Highest number of passengers who DID NOT survive were from 3rd class.


```python
df[['sex', 'survived']].groupby(['sex'], as_index=False).mean()
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
      <th>sex</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.738832</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.215886</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(x='sex',hue="survived", data=df)
plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_27_0.png)


> Survival rate of female passengers was more than male


```python
age_survived = sns.FacetGrid(df, col='survived')
age_survived.map(plt.hist, 'age', bins=15)
plt.show()
```


![png]({{ site.url }}{{ site.baseurl }}/images/dlithe-titanic/DLithe-ASSIGNMENT-titanic_29_0.png)


>Older passengers (of around 80 years old) survived.

<h2>Machine Learning Model - Logistic Regression</h2>


```python
# Convert Boolean to Integer

df["alone"] = df["alone"].astype(int)
df.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>who</th>
      <th>embark_town</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>man</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>woman</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>woman</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>man</td>
      <td>Southampton</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encode the data (convert strings to numbers) so that the model can understand.

le_sex = LabelEncoder()
df["sex"]=le_sex.fit_transform(df["sex"])

le_who = LabelEncoder()
df["who"]=le_who.fit_transform(df["who"])

le_embark_town = LabelEncoder()
df["embark_town"]=le_embark_town.fit_transform(df["embark_town"])
```


```python
print(df.shape)
df.head(10)
```

    (782, 10)
    




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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>who</th>
      <th>embark_town</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>51.8625</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>11.1333</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>30.0708</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split the dependent and independent values
x = df.drop("survived", axis=1)
y = df["survived"]
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

    Train set: (625, 9) (625,)
    Test set: (157, 9) (157,)
    


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

    Mean Absolute Error:  0.17834394904458598
    Accuracy:  0.821656050955414
    

### Highest Accuracy : 0.821656050955414
