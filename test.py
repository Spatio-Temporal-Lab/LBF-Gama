import pandas as pd
from sklearn.preprocessing import StandardScaler
train_data=pd.read_csv('dataset/train.csv')
test_data=pd.read_csv('dataset/test.csv')
data = pd.concat([train_data, test_data])

data.dropna(inplace=True)
data=data[data['rowv']!=-9999]
data=data[data['colv']!=-9999]
data=data[data['expRad_u']!=0.0]
data=data[data['expRad_g']!=0]
data=data[data['expRad_r']!=0]
data=data[data['expRad_i']!=0]
data=data[data['expRad_z']!=0]

data['type']=data['type'].replace({'star': 1, 'galaxy': 0})

true_data=data[data['type']==1]
false_data=data[data['type']==0]
true_data=true_data.sample(frac=0.9, random_state=42)

train_true=true_data.sample(frac=0.8, random_state=42)
test_true = true_data.drop(train_true.index)

query_data=false_data.sample(482213)
false_data=false_data.drop(query_data.index)
train_false=false_data.sample(frac=0.8, random_state=42)
test_false=false_data.drop(train_false.index)
train_data=pd.concat([train_true, train_false])
test_data=pd.concat([test_true, test_false])
train_data.to_csv('Train_COD.csv')
test_data.to_csv('Test_COD.csv')
query_data.to_csv('Query_COD.csv')