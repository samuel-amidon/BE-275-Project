import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pickle

from tensorly.decomposition import parafac
from tensorly.tenalg import outer
from sklearn.decomposition import PCA

from pseudobulkFuncs import pseudobulk, normalize, prune_genes
from tensorFactorizationFuncs import tensor_restructure, reconstruction_error

# Notes
#  adat.raw.X is the same as adat.X



## Read and process raw data
# adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/VUMC_HTAN_DIS_EPI_V2.h5ad")

# pseudobulk_adat = pseudobulk(adat, min_expression=0.01)

# ad.AnnData.write(pseudobulk_adat, "/Users/samuelamidon/Desktop/BE275 Project Supp/pseudobulk_adat.h5ad")

# print(pseudobulk_adat)
# print(pseudobulk_adat.X)
# print(pseudobulk_adat.obs)
# print(pseudobulk_adat.var)



## Restructure processed data
pseudobulk_adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/processed_0.01cutoff.h5ad")

tensor_data = tensor_restructure(pseudobulk_adat)
# print(tensor_data)
# print(tensor_data.shape)



## Test tensor factorization
# cp = parafac(tensor_data, rank=9, verbose=0, n_iter_max=100)

# print(cp[0])
# print(cp[1])



## Test reconstruction
# a, m = reconstruction_error(tensor_data, cp, manual_check=True)

# print(a)
# print(m)



## Run factorization with all ranks, save results
# cp_ls = []
# err_ls = []
# rank = 9
# for i in range(rank):
#     cp_ls.append(parafac(tensor_data, rank=i+1))
#     err_ls.append(reconstruction_error(tensor_data, cp_ls[i]))

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/cp_ls.pkl", "wb") as f:
#     pickle.dump(cp_ls, f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/err_ls.pkl", "wb") as f:
#     pickle.dump(err_ls, f)



## Plot reconstruction error vs component number
with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/cp_ls.pkl", "rb") as f:
    cp_ls = pickle.load(f)

with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/err_ls.pkl", "rb") as f:
    err_ls = pickle.load(f)

# plt.plot(range(1,10), err_ls, marker="o")
# plt.xlabel("Component Number")
# plt.ylabel("Reconstruction Error")
# plt.show()



## Compare with PCA
CP_pd = pd.DataFrame(columns = ["Size", "Error"])
rank=9

for i in range(rank):
    CP_pd.loc[len(CP_pd)+1] = [(i+1) * np.sum(tensor_data.shape), err_ls[i]]


PCA_pd = pd.DataFrame(columns = ["Size", "Error"])
flatten = tensor_data.reshape(tensor_data.shape[0], -1)
pca = PCA(n_components=flatten.shape[0])
pca.fit(flatten)
pca_errors = 1 - np.cumsum(pca.explained_variance_ratio_)
pca_sizes = np.sum(flatten.shape) * np.arange(1, flatten.shape[0] + 1)
for i in range(rank):
    PCA_pd.loc[len(PCA_pd)+1] = [pca_sizes[i], pca_errors[i]]

plt.plot(CP_pd["Size"], CP_pd["Error"], "o-", label="CP")
plt.plot(PCA_pd["Size"], PCA_pd["Error"], "o--", label="PCA")
plt.xlabel("Component Size")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.show()