# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/90ffc28e-e70e-442d-8ab5-5996097a59fc)


```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/94a01983-60fd-446b-8853-d520e8f89db6)


```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/b994eb2a-ece1-448f-85c5-22ac7f5f6c7f)


```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/43026283-38f9-4d83-a01c-ae58f98d91db)


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/4f363047-3ea0-402f-bfe8-81ad7ac77144)


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/b7cb0647-1f7f-4d54-80d9-3ef2ef83e487)


```
data2
```
![image](https://github.com/user-attachments/assets/a91f32d0-9ac1-4e33-89e6-ba4746d9b809)


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/a6d9a2f7-3a28-4da4-b5c1-f01d499b8390)


```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/b8016569-ea16-451f-ba56-b8b6119b60db)


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/5f81c9ad-d292-436b-ae1c-7a54b2789acf)


```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/a6bce1e4-e078-46bd-b5af-9358c554b733)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/1395bd81-829c-40e5-b0b6-20f2842ea4f8)


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/7b567b8c-6758-4eee-a6f6-c22240c18543)


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/4d7e2c85-c663-4f4d-afae-1f3cfd90f7d6)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/3b57b3b6-cbe6-4a59-9963-3881c2726d1f)


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/2ffa2507-8a4d-41ec-bd94-39cbae28ed18)


```
data.shape
```
![image](https://github.com/user-attachments/assets/f2085d63-5216-4945-8460-bf42bc07e88c)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/3b8ded62-5454-4e2c-a28f-99e2e69c5070)


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/f84e9431-3fb8-46ed-8341-79161dd05ebc)


```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/5c1b560b-4bae-44ff-bffa-40adabd10a3b)


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/e7716975-5071-4e11-b550-c55f5ba00fa9)


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/678ccd3c-dc5a-4397-a61f-5b961ec69444)

# RESULT:
      
Thus the code to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is implemented successfully.
