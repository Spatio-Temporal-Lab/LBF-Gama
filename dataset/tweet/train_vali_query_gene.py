import pandas as pd

data = pd.read_csv('tweet.csv')
fake_data = pd.read_csv('fake_data_tweet.csv')
data['is_in'] = 1
fake_data['is_in'] = 0
length = len(data) * 0.01
train_true = data.iloc[:int(length), :]
train_fake = fake_data.iloc[:int(length)*2, :]
train_data = pd.concat([train_true, train_fake])
query_fake = fake_data.iloc[int(length)*2:, :].sample(100000)
val_data = data.iloc[int(length):, :]
query_true = data.sample(100000, random_state=0)
query_data = pd.concat([query_true, query_fake])
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('vali_data.csv', index=False)
query_data.to_csv('query_data.csv', index=False)
