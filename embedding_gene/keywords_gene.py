import gensim
import csv
import numpy as np
import pandas as pd
from collections import Counter
import pickle

# add random vectors of unknown words which are not in pre-trained vector file.
# if pre-trained vectors are not used, then initialize all words in vocab with random value.

'''
spell = SpellChecker()
def add_unknown_words(word_vecs, vocab,k):
    cnt=0
    for word in vocab:
        if word not in word_vecs.keys():
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k) 
            cnt=cnt+1
    print('cnt:',cnt)

'''
# 加载Google's word2vec模型
model = gensim.models.KeyedVectors.load_word2vec_format('../embedding_gene/GoogleNews-vectors-negative300.bin',
                                                        binary=True)
# print('girl' in model)
# 从文件中读取单词列表

'''

chunksize = 100000
chunk_no = 0
word_list = []
# output_file = "all_keywords.txt"
word_cnt = 0
missing_cnt = 0
# with open(output_file, "a") as f:
t = 0
for chunk in pd.read_csv("../dataset/tweet_clean.csv", chunksize=chunksize):
    print("处理到块:", chunk_no)
    chunk_no += 1
    #chunk.dropna(inplace=True)
    keywords = chunk['1'].str.split().explode().tolist()
    for keyword in keywords:
        if type(keyword) == float:
            print(keyword)
        if keyword not in model:
            # print(keyword)
            # f.write(keyword.lower() + ",")
            missing_cnt += 1
        else:
            word_cnt += 1

    # word_list=word_list+[keyword.lower() for keyword in keywords]
    #print(word_cnt)
    #print(missing_cnt)

#print("关键字已成功写入到文件:", output_file)




# 定义关键词文件路径
keywords_file = "all_keywords.txt"

# 读取关键词文件并将关键词存储到列表中
with open(keywords_file, "r") as f:
    # 读取文件内容并使用逗号分隔
    keywords_text = f.read()
    
    # 将分隔后的关键词存储到列表中
    keywords_list = keywords_text.split(",")

#print("读取的关键词列表:", keywords_list)


word_count = Counter(keywords_list)
tf = list(word_count.items())
max_freq = max(tf, key=lambda x: x[1])[1]
tf = sorted(tf, key=lambda x: x[1])
print(word_count)
print(tf)
print(len(tf))
vocab=[]
wrong_keyword=[]
for t in tf:
    if t[1]>=5:
        vocab.append(t[0])
    else:
        wrong_keyword.append(t[0])
vocab=list(set(vocab))
print(len(vocab))
print(len(wrong_keyword))
correct_dict={}
for word in wrong_keyword:
    corrected_word=spell.correction(word)
    print(word,corrected_word)
    
    if word != corrected_word:
        correct_dict[word]=corrected_word

with open('my_dict.pkl', 'wb') as f:
    pickle.dump(correct_dict, f)

vectors = {}
cnt=0
for word in vocab:
    if word in model.key_to_index.keys():
        lenth = model[word].size
        vectors[word] = model[word]
    else:
        cnt+=1

lenth=300
add_unknown_words(vectors,vocab,lenth)

for word in vectors.keys():
    if (len(vectors[word]) != lenth):
        # print(vectors[word])
        print(len(vectors[word]))
        print("not equal2")
# print(vectors)
# 将结果存储到csv文件中


with open('all_keywords_embedding.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['word', 'vector'])
    for key,item in vectors.items():
        writer.writerow([key, item])
        # cnt+=1
        # print(cnt)
'''

keywords = []
'''
with open('validation_set_region.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        keyword = row[1]
        words = keyword.split()
        words = [word.lower() for word in words]  # 转换为小写
        keywords.extend(words)

vocab = list(set(keywords))

lenth = 0
'''
keywords = pd.read_csv("../dataset/tweet_clean.csv")
keywords = keywords.drop(keywords.columns[[0, 1]], axis=1)
keywords.columns = ['timestamp', 'keywords', 'lat', 'lon']
keywords.to_csv('tweet_clean_again.csv', index=False)
'''
keywords = keywords['1'].str.split().explode().tolist()
keywords = [keyword.lower() for keyword in keywords]

vocab = list(set(keywords))

print(vocab)
print(len(vocab))

vectors = {}
for word in vocab:
    if word in model.key_to_index.keys():
        vectors[word] = model[word]

print(len(vectors))
'''
'''
for word in vectors.keys():
    if (len(vectors[word]) != lenth):
        # print(vectors[word])
        print(len(vectors[word]))
        print("not equal2")
# print(vectors)
# 将结果存储到csv文件中


with open('tweet_keywords_embedding.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['word', 'vector'])
    for key, item in vectors.items():
        writer.writerow([key, item])
        # cnt+=1
        # print(cnt)
'''
