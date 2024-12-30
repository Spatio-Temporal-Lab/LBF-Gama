import pandas as pd

# 将标签（1）和标签（0）分别过滤出来
df_train = pd.read_csv('url_train.csv')
df_test = pd.read_csv('url_test.csv')

df_1 = pd.concat([df_train[df_train['url_type'] == 1], df_test[df_test['url_type'] == 1]])
df_0 = pd.concat([df_train[df_train['url_type'] == 0], df_test[df_test['url_type'] == 0]])

frac = 0.1

df_1_sample = df_1.sample(frac=frac, random_state=42)
df_0_sample = df_0.sample(frac=frac, random_state=42)
df_combined = pd.concat([df_1_sample, df_0_sample])
df_combined = df_combined.sample(frac=1, random_state=42)
df_combined.to_csv('url_sample' + str(frac) + '.csv', index=False)