from lib.data_processing import *
import lib.network

dataset = "tweet"
input_dim = 372
output_dim = 1
all_memory = 5 * 1024 * 1024
all_record = 10767160 + 30000
learning_rate = 0.005
hidden_units = (8, 512)
bf_name = "best_bloom_filter.pkl"


word_dict, region_dict = loading_embedding(dataset)
train_loader, val_loader = loading_data(word_dict=word_dict, region_dict=region_dict, dataset=dataset,
                                        dataset_type="train")
nas_opt = lib.network.Bayes_Optimizer(input_dim=input_dim, output_dim=output_dim, train_loader=train_loader,
                                      val_loader=val_loader, learning_rate=learning_rate, hidden_units=hidden_units)
model = nas_opt.optimize()
torch.save(model, 'best_tweet_model.pth')

# 如果没法直接跑全程 可以先保存训练完的模型再加载
# model = torch.load('best_tweet_model.pth')

model.eval()

lib.network.train(model, train_loader=train_loader, val_loader=val_loader)
lib.network.validate(model, region_dict=region_dict, word_dict=word_dict, dataset=dataset)
bloom_filter = lib.network.create_bloom_filter(dataset=dataset, bf_name=bf_name)
lib.network.query(model, region_dict=region_dict, word_dict=word_dict, dataset=dataset, bloom_filter=bloom_filter)
