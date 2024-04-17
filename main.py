from lib.data_processing import *
import lib.data_processing as dp
import lib.network

dataset = "tweet"
input_dim = 372
output_dim = 1
all_memory = 5 * 1024 * 1024
all_record = 10767160 + 30000
word_dict, region_dict = loading_embedding(dataset)
validation_loader = loading_data(word_dict=word_dict, region_dict=region_dict, dataset=dataset,
                                 dataset_type="vali")
'''
train_loader, val_loader = loading_data(word_dict=word_dict, region_dict=region_dict, dataset=dataset,
                                        dataset_type="train")

NAS_opt = lib.network.Bayes_Optimizer(input_dim=input_dim, output_dim=output_dim, train_loader=train_loader,
                                      val_loader=val_loader, learning_rate=0.005, hidden_units=(8, 512))
model = NAS_opt.optimize()
# model = lib.network.SimpleNetwork(structure=[128], input_dim=input_dim, output_dim=output_dim)

lib.network.train(model, train_loader=train_loader, val_loader=val_loader)
# lib.network.validate(model,train_loader=train_loader,val_loader=val_loader)
'''
