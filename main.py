from lib.data_processing import *
import lib.data_processing as dp
import lib.network

dataset="tweet"
input_dim=372
output_dim=1

word_dict,region_dict=loading_embedding(dataset)
train_loader,val_loader=loading_data(word_dict=word_dict,region_dict=region_dict,dataset=dataset,dataset_type="train")
model=lib.network.SimpleNetwork(structure=[12],input_dim=input_dim,output_dim=output_dim)
lib.network.train(model,train_loader=train_loader,val_loader=val_loader)
#lib.network.validate(model,train_loader=train_loader,val_loader=val_loader)
