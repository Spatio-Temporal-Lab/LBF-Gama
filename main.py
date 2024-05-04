from lib.data_processing import *
import lib.network
import pickle

dataset = "yelp"
input_dim = 372
output_dim = 1
all_memory = 634 * 1024         # tweet模型大小：5 * 1024 * 1024
all_record = 33094122
learning_rate = 0.005
hidden_units = (8, 512)
bf_name = "best_bloom_filter_yelp.pkl"

word_dict, region_dict = loading_embedding(dataset)
train_loader, val_loader = loading_data(word_dict=word_dict, region_dict=region_dict, dataset=dataset,
                                        dataset_type="train")
# #
nas_opt = lib.network.Bayes_Optimizer(input_dim=input_dim, output_dim=output_dim, train_loader=train_loader,
                                      val_loader=val_loader, learning_rate=learning_rate, hidden_units=hidden_units, all_record=all_record)
model = nas_opt.optimize()
print("has optimized")
lib.network.train(model, train_loader=train_loader, val_loader=val_loader, all_record=all_record, all_memory=all_memory)
torch.save(model, 'best_yelp_model.pth')



# 如果没法直接跑全程 可以先保存训练完的模型再加载
model = torch.load('best_yelp_model.pth')
print("have loaded")
model.eval()

#获得学习模型的内存大小
model_size = lib.network.get_model_size(model)
bloom_size = all_memory - model_size
#
lib.network.validate(model, region_dict=region_dict, word_dict=word_dict, dataset=dataset)
bloom_filter = lib.network.create_bloom_filter(dataset=dataset, bf_name=bf_name,bf_size=bloom_size)
# bloom_filter = lib.network.create_bloom_filter(dataset=dataset, bf_name=bf_name)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)


#load
# def load_bloom_filter(file_path):
#     with open(file_path, 'rb') as f:
#         bloom_filter = pickle.load(f)
#     return bloom_filter
# bloom_filter = load_bloom_filter(bf_name)
lib.network.query(model, region_dict=region_dict, word_dict=word_dict, dataset=dataset, bloom_filter=bloom_filter)

