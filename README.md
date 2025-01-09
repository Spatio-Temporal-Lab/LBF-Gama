# Gama: General Adaptive Memory Allocation for Learned Bloom Filters
***
Gama is a holistic framework that solves the memory allocation problem of LBFs. The objective of Gama is
to minimize the overall FPR of an LBF under a given memory
budget

### Prerequisites for testing
- Default Python version used in experiments: 3.10.12
- Run model_generate.py to generate a fixed-size LightGBM model for method comparison
- Run dataset/get_sample.py to generate the sampled dataset

### Run
- We use URL Dataset as the default for testing

| Method Name      | Python Script Path      |
| ---------------- |-------------------------|
| Gama-LBF         | lgb_url_GamaLBF_ada.py  |
| Gama-PLBF        | lgb_url_GamaPLBF_ada.py |
| PLBF             | PLBF/main.py            |
| Ada-BF           | ada-bf/main.py          |
| Sandwich-BF      | sandwich-lbf/main.py    |
| LBF              | lbf/main.py             |
| BF               | bf_url_main.py          |


### Datasets

The datasets used in paper can be downloaded from the following links:

- URL：https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
- Yelp: https://www.yelp.com/dataset/download
- COD: https://www.kaggle.com/datasets/hari31416/celestialclassify


