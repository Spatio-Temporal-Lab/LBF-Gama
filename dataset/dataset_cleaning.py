import pandas as pd

# 读取数据
data = pd.read_csv("tweet.csv")
print(data)
data=data.dropna()
print(data)
data['2'] = pd.to_numeric(data['2'], errors='coerce')
data = data.dropna(subset=['2'])
data['2'] = data['2'].astype(float)

data['3'] = pd.to_numeric(data['3'], errors='coerce')
data = data.dropna(subset=['3'])
data['3'] = data['3'].astype(float)
print(data)
def is_alpha(s):
    try:
        for char in s:
            a=ord(char)
            if not(a>20 and a<126):
                return False
        return True
    except: 
        return False
data = data[data['1'].apply(is_alpha)]
print(data)

data=data[(((data['3']>-120) & (data['3']<-74)))]
data=data[(((data['2']>27) & (data['2']<54)))]

print(data)
data.to_csv("tweet.csv")