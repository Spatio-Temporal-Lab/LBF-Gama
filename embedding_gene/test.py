import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from spellchecker import SpellChecker

spell = SpellChecker()

# 定义一个函数来读取并处理 TXT 文件
def process_txt_file(file_path):
    with open(file_path, "r") as f:
        keywords_text = f.read()
        # 将分隔后的关键词存储到列表中
        keywords_list = keywords_text.split(",")
        return keywords_list

# 定义一个函数来进行拼写检查
def spellcheck(word):
    corrected_word = spell.correction(word)
    if corrected_word != word:
        correction_dict[word] = corrected_word
    if corrected_word is None:
        candidates = spell.candidates(word)
        if candidates:
            return candidates[0]  # 返回候选词列表中的第一个单词
        else:
            return word  # 如果没有候选词，则返回原单词
    else:
        return corrected_word

# 单个 TXT 文件路径
txt_file = "all_keywords.txt"

# 读取并处理单个 TXT 文件
processed_text = process_txt_file(txt_file)

# 创建线程池
with ThreadPoolExecutor(max_workers=20) as executor:
    # 提交任务并获取 Future 对象列表
    futures = [executor.submit(spellcheck, word) for word in processed_text]

    # 初始化计数器
    processed_count = 0

    # 处理完成的任务数量
    for future in as_completed(futures):
        result = future.result()
        # 在这里处理结果
        processed_count += 1
        print(f"Processed {processed_count}/{len(processed_text)} words.")

# 构建映射字典
correction_dict = {}

# 保存映射字典
with open('correction_dict.pkl', 'wb') as f:
    pickle.dump(correction_dict, f)
