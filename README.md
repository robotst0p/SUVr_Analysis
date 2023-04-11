# Catalysis-Clustering
This repository contains code for 2020 [paper "Catalysis Clustering with GAN by Incorporating Domain Knowledge"](https://dl.acm.org/doi/10.1145/3394486.3403187)

## Catalysis generation with GAN

WGAN training procedure follows [this repo](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master)

The main code for this stage is the following:
-  `lib` folder contains model builders and utility functions
-  `wgan-luad.py` contains training and generation procedures for lung somatic mutation profiles
-  `wgan-ov.py` contains training and generation procedures for ovarian somatic mutation profiles
-  `requirements.txt` contains the list of requirements for this code
    - You can install requirements all together with: `pip install -r requirements.txt` 
- It requires preprocessed mutation profiles as an input, e.g. `wgan-ov.py` line 118: `x_train = np.load("ov_genes_converted.npy");`
    - This should be changed to accomodate the dataset, which will be used with this code

### Usage

After all requirements are installed:
1. Add `"ov_genes_converted.npy"` or `"luad_genes_converted.npy"` to the same folder with `python wgan-*.py`, where `*` is either `luad` or `ov`
2. Run `python wgan-*.py` to train WGAN, where `*` is either `luad` or `ov`
3.Run `python wgan-*.py -g <PATH/TO/GENERATOR/WEIGHTS>` to generate synthetic data, where `*` is either `luad` or `ov`

## Evaluation
Please note: this code is for the task of clustering evaluation through survival analysis and SCM measure, described in the paper.

`evaluation` folder contains R code for survival analysis and SCM measure computation, which results were presented in the paper:
- `lung_cluster_evaluation.R` contains sample code for analysing only one file with survival information and clustering assignment
- `ov_cluster_evaluation.R` contains sample code for analysing a set of files with survival information and clustering assignment
    - User is required to specify paths to folder, containing survival info and folders where to store images and R data
- Please refer to [R Project](https://www.r-project.org/) for further assistance with an R code

## Citation

```
@inproceedings{10.1145/3394486.3403187,
author = {Andreeva, Olga and Li, Wei and Ding, Wei and Kuijjer, Marieke and Quackenbush, John and Chen, Ping},
title = {Catalysis Clustering with GAN by Incorporating Domain Knowledge},
year = {2020},
isbn = {9781450379984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394486.3403187},
doi = {10.1145/3394486.3403187},
abstract = {Clustering is an important unsupervised learning method with serious challenges when data is sparse and high-dimensional. Generated clusters are often evaluated with general measures, which may not be meaningful or useful for practical applications and domains. Using a distance metric, a clustering algorithm searches through the data space, groups close items into one cluster, and assigns far away samples to different clusters. In many real-world applications, the number of dimensions is high and data space becomes very sparse. Selection of a suitable distance metric is very difficult and becomes even harder when categorical data is involved. Moreover, existing distance metrics are mostly generic, and clusters created based on them will not necessarily make sense to domain-specific applications. One option to address these challenges is to integrate domain-defined rules and guidelines into the clustering process. In this work we propose a GAN-based approach called Catalysis Clustering to incorporate domain knowledge into the clustering process. With GANs we generate catalysts, which are special synthetic points drawn from the original data distribution and verified to improve clustering quality when measured by a domain-specific metric. We then perform clustering analysis using both catalysts and real data. Final clusters are produced after catalyst points are removed. Experiments on two challenging real-world datasets clearly show that our approach is effective and can generate clusters that are meaningful and useful for real-world applications.},
booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
pages = {1344–1352},
numpages = {9},
keywords = {clustering evaluation, cancer subtyping, GAN, domain-informed clustering},
location = {Virtual Event, CA, USA},
series = {KDD '20}
}
```
