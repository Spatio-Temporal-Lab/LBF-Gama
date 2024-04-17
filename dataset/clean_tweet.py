import pandas as pd
import gensim

# Step 1: Read keywords from the TXT file
with open('all_keywords.txt', 'r') as file:
    txt_keywords = file.read().strip().split(',')

# Convert the keyword list to a set for quick lookup
keyword_set = set(txt_keywords)

# Step 2: Read the CSV file
# Assuming the third column is named 'Keywords'
csv_data = pd.read_csv('tweet.csv')

model = gensim.models.KeyedVectors.load_word2vec_format('../embedding_gene/GoogleNews-vectors-negative300.bin',
                                                        binary=True)


# Step 3: Define a function to remove keywords
def remove_keywords(keywords):
    # Split the keywords, filter out the unwanted ones, and rejoin them
    keyword_list = keywords.split()
    filtered_keywords = [word.lower() for word in keyword_list if word.lower() in model]
    if len(keyword_list) != len(filtered_keywords):
        print("origin list:", keyword_list)
        print("filtered list:", filtered_keywords)
    return ' '.join(filtered_keywords)


# Apply this function to the 'Keywords' column
csv_data['1'] = csv_data['1'].apply(remove_keywords)

# Step 4: Save the modified CSV file
csv_data=pd.read_csv('tweet_clean.csv')
print(csv_data)
csv_data.dropna(inplace=True)
print(csv_data)
csv_data.to_csv('tweet_clean.csv', index=False)
