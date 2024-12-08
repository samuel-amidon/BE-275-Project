import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib
from matplotlib import gridspec, pyplot as plt
import pickle

from tensorly.decomposition import parafac
from tensorly.tenalg import outer
from sklearn.decomposition import PCA, NMF
from figures.factor_plots import makeFigure

from pseudobulkFuncs import pseudobulk, normalize, prune_genes
from tensorFactorizationFuncs import tensor_restructure, reconstruction_error

# Notes
#  adat.raw.X is the same as adat.X
#  "Patient" and "Tumor_Type" observations are all NaN?



## Read and process raw data
# adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/VUMC_HTAN_DIS_EPI_V2.h5ad")

# print(adat.obs.keys())
# print(adat.obs_names)
# print(adat.obs["HTAN Specimen ID"].unique())
# print(adat.obs["Cell_Type"].unique())
# print(adat.obs["B cell"].unique())
# print(adat.obs["Patient"].unique())
# print(adat.obs["Sample_Classification"].unique())
# print(adat.obs["Macrophage"].unique())
# print(adat.obs["B cell"])
# print(adat.obs["Macrophage"])

# pseudobulk_adat = pseudobulk(adat, min_expression=0.01)

# ad.AnnData.write(pseudobulk_adat, "/Users/samuelamidon/Desktop/BE275 Project Supp/pseudobulk_adat.h5ad")

# print(pseudobulk_adat)
# print(pseudobulk_adat.X)
# print(pseudobulk_adat.obs)
# print(pseudobulk_adat.var)



## Restructure processed data
pseudobulk_adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/processed_0.01cutoff.h5ad")

tensor_data, samples, genes, cell_types, names = tensor_restructure(pseudobulk_adat)



## Test tensor factorization
# cp = parafac(tensor_data, rank=9, verbose=0, n_iter_max=100, normalize_factors=True)

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



## Load CPD results
# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/cp_ls.pkl", "rb") as f:
#     cp_ls = pickle.load(f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/err_ls.pkl", "rb") as f:
#     err_ls = pickle.load(f)



## Plot reconstruction error vs component number
# plt.plot(range(1,10), err_ls, marker="o")
# plt.xlabel("Component Number")
# plt.ylabel("Reconstruction Error")
# plt.show()



## Perform NMF
# nmf_ls = []
# nmf_err_ls = []
# rank = 13
# for i in range(rank):
#     nmf = NMF(n_components=i+1)
#     nmf_data = tensor_data.reshape(tensor_data.shape[0], -1)
#     nmf.fit(nmf_data)
#     nmf_ls.append(nmf)
#     # nmf_err_ls.append(nmf.reconstruction_err_)
#     nmf_err_ls.append(np.sum(nmf_data - nmf.inverse_transform(nmf.transform(nmf_data)))**2 / np.sum(nmf_data**2))

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/nmf_ls.pkl", "wb") as f:
#     pickle.dump(nmf_ls, f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/nmf_err_ls.pkl", "wb") as f:
#     pickle.dump(nmf_err_ls, f)



## Compare with PCA (and NMF)
# CP_pd = pd.DataFrame(columns = ["Size", "Error"])
# rank=9

# for i in range(rank):
#     CP_pd.loc[len(CP_pd)+1] = [(i+1) * np.sum(tensor_data.shape), err_ls[i]]


# PCA_pd = pd.DataFrame(columns = ["Size", "Error"])
# flatten = tensor_data.reshape(tensor_data.shape[0], -1)
# pca = PCA(n_components=flatten.shape[0])
# pca.fit(flatten)
# pca_errors = 1 - np.cumsum(pca.explained_variance_ratio_)
# pca_sizes = np.sum(flatten.shape) * np.arange(1, flatten.shape[0] + 1)
# for i in range(flatten.shape[0]-1):
#     PCA_pd.loc[len(PCA_pd)+1] = [pca_sizes[i], pca_errors[i]]


# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/nmf_ls.pkl", "rb") as f:
#     nmf_ls = pickle.load(f)
# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/nmf_err_ls.pkl", "rb") as f:
#     nmf_err_ls = pickle.load(f)
# NMF_pd = pd.DataFrame(columns = ["Size", "Error"])
# nmf_sizes = np.sum(flatten.shape) * np.arange(1, len(nmf_err_ls) + 1)
# for i in range(len(nmf_err_ls)):
#     NMF_pd.loc[len(NMF_pd)+1] = [nmf_sizes[i], nmf_err_ls[i]]

# plt.plot(CP_pd["Size"], CP_pd["Error"], "o-", label="CP")
# plt.plot(PCA_pd["Size"], PCA_pd["Error"], "o--", label="PCA")
# plt.plot(NMF_pd["Size"], NMF_pd["Error"], "o-.", label="NMF")
# plt.xlabel("Component Size")
# plt.ylabel("Reconstruction Error")
# plt.legend()
# plt.xlim(0, 1.5e6)
# plt.ylim(0, 4)
# plt.show()



## Used 6 components...
cp = parafac(tensor_data, rank=6, verbose=0, n_iter_max=100, normalize_factors=True)



## Plot CPD results
makeFigure(cp, samples.tolist(), genes.index.tolist(), cell_types.tolist(), names, 0.02)
