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
from figures.factor_plots import factorHeatmaps
from figures.gene_distributions import geneDistributions
from figures.reconstruction_error_plot import errorPlot
from figures.analysis_comparison import analysisComparison
from figures.plot_raw_expressions import meanExpression

from pseudobulkFuncs import pseudobulk, normalize, prune_genes
from tensorFactorizationFuncs import tensor_restructure, reconstruction_error

# Notes
#  adat.raw.X is the same as adat.X
#  "Patient" and "Tumor_Type" observations are mostly NaN?
#  Need to filter mitochondrial genes
#  Long non coding RNA genes should also be removed...

#  NL: normal mucosal
#  AD: adenoma
#  HP: hyperplastic polyp
#  SSL: sessile serrated lesion
#  TA: tubular adenoma
#  TVA: tubulovillous adenoma
#  SER: serrated adenoma

# GOB: goblet cells?
# TAC: transient amplifying cells?
# ASC: adenosquamous carcinoma cells?
# TUF: fibroblasts???
# ABS: ???
# STM: stem cells?
# SSC: ???
# SCC: squamous cell carcinoma cells
# EE: enteroendocrine cells?
# CT: ???



## Read and process raw data
# adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/VUMC_HTAN_DIS_EPI_V2.h5ad")

# pseudobulk_adat = pseudobulk(adat, min_expression=0.01)

# ad.AnnData.write(pseudobulk_adat, "/Users/samuelamidon/Desktop/BE275 Project Supp/pseudobulk_adat.h5ad")



## Restructure processed data
# pseudobulk_adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/processed_0.01cutoff_v1.h5ad")
# pseudobulk_adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/processed_0.01cutoff_v2.h5ad")
pseudobulk_adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/pseudobulk_adat.h5ad")
# print(pseudobulk_adat.obs)



# adat.write_csvs("/Users/samuelamidon/Desktop/BE275 Project Supp/")
# pseudobulk_adat.write_csvs("/Users/samuelamidon/Desktop/BE275 Project Supp/")

# pd.DataFrame(pseudobulk_adat.X).to_csv("/Users/samuelamidon/Desktop/BE275 Project Supp/pseudobulk_data.csv")


# print((adat.obs["Polyp_Type"][adat.obs["HTAN Specimen ID"] == "HTA11_5212_2000002011"]))
# print(all(adat.obs["Polyp_Type"][adat.obs["HTAN Specimen ID"] == "HTA11_5212_2000002011"])=="NL")

# print((pseudobulk_adat.obs["Polyp_Type"][pseudobulk_adat.obs["HTAN Specimen ID"] == "HTA11_5212_2000002011"]))
# print(all(pseudobulk_adat.obs["Polyp_Type"][pseudobulk_adat.obs["HTAN Specimen ID"] == "HTA11_5212_2000002011"])=="NL")



## Filter bad genes
genes_to_remove = ["LINC00486","MT-ATP6","MT-CO1","MT-CO2","MT-CO3","MT-CYB","MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-RNR1","MT-RNR2","RP5-857K21.4"]
# genes_to_remove = ["MT-ATP6","MT-CO1","MT-CO2","MT-CO3","MT-CYB","MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-RNR1","MT-RNR2"]
# genes_to_remove = []
pseudobulk_adat_filt = pseudobulk_adat[:, ~pseudobulk_adat.var_names.isin(genes_to_remove)]

samples_to_remove = ["HTA11_5212_2000002011", "HTA11_83_2000002011"]
pseudobulk_adat_filt = pseudobulk_adat_filt[~pseudobulk_adat_filt.obs["HTAN Specimen ID"].isin(samples_to_remove)]



## Restructure filtered data
tensor_data_filt = tensor_restructure(pseudobulk_adat_filt)
tensor_data = tensor_data_filt

samples = pseudobulk_adat_filt.obs["HTAN Specimen ID"].unique()
genes = pseudobulk_adat_filt.var
cell_types = pseudobulk_adat_filt.obs["Cell_Type"].unique()
names = ["Samples", "Genes", "Cell Types"]

classification = [
        pseudobulk_adat_filt.obs["Sample_Classification"][pseudobulk_adat_filt.obs["HTAN Specimen ID"] == sample].iloc[0]
        for sample in samples
    ]
polyp_type = [
        pseudobulk_adat_filt.obs["Polyp_Type"][pseudobulk_adat_filt.obs["HTAN Specimen ID"] == sample].iloc[0]
        for sample in samples
    ]



## Test tensor factorization
# cp = parafac(tensor_data, rank=9, verbose=0, n_iter_max=100, normalize_factors=True)



## Test reconstruction
# a, m = reconstruction_error(tensor_data, cp, manual_check=True)



## Run factorization with all ranks, save results
# cp_ls = []
# err_ls = []
# rank = 9
# for i in range(rank):
#     cp_ls.append(parafac(tensor_data, rank=i+1, normalize_factors=True, init="random"))
#     err_ls.append(reconstruction_error(tensor_data, cp_ls[i]))

# cp_ls = [[], [], []]
# err_ls = [[], [], []]
# rank = 20
# for i in range(3):
#     for j in range(rank):
#         cp_ls[i].append(parafac(tensor_data, rank=j+1, normalize_factors=True, init="random"))
#         err_ls[i].append(reconstruction_error(tensor_data, cp_ls[i][j]))

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/cp_ls_20.pkl", "wb") as f:
#     pickle.dump(cp_ls, f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/err_ls_20.pkl", "wb") as f:
#     pickle.dump(err_ls, f)



## Load CPD results
# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/randoms/cp_ls_svd.pkl", "rb") as f:
#     cp_ls_svd = pickle.load(f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/randoms/err_ls_svd.pkl", "rb") as f:
#     err_ls_svd = pickle.load(f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/randoms/cp_ls_1.pkl", "rb") as f:
#     cp_ls_ran = pickle.load(f)

# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/randoms/err_ls_1.pkl", "rb") as f:
#     err_ls_ran = pickle.load(f)

# err_opt = err_ls_ran[0]
# for i in range(9):
#     err_opt[i] = min([err_ls_ran[j][i] for j in range(4)])

with open("/Users/samuelamidon/Desktop/BE275 Project Supp/cp_ls_20.pkl", "rb") as f:
    cp_ls = pickle.load(f)

with open("/Users/samuelamidon/Desktop/BE275 Project Supp/err_ls_20.pkl", "rb") as f:
    err_ls = pickle.load(f)



## Plot reconstruction error vs component number
# errorPlot(err_ls_ran, overlay_svd=True, svd_errors=err_ls_svd)
# errorPlot(err_ls_svd)
# errorPlot(err_opt)
errorPlot(err_ls)



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
# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/nmf_ls.pkl", "rb") as f:
#     nmf_ls = pickle.load(f)
# with open("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/nmf_err_ls.pkl", "rb") as f:
#     nmf_err_ls = pickle.load(f)

# # analysisComparison(err_ls, tensor_data, nmf=True, nmf_errors=nmf_err_ls)
# analysisComparison(err_ls, tensor_data)



## Find best random states
# for i in np.arange(0, 10):
#     cp5 = parafac(tensor_data, rank=5, normalize_factors=True, random_state=int(i), init="random")
#     print("random state", i, reconstruction_error(tensor_data, cp5)) #Use random state 4
    # cp6 = parafac(tensor_data, rank=6, verbose=0, n_iter_max=100, normalize_factors=True, random_state=int(i), init="random")
    # print("random state", i, reconstruction_error(tensor_data, cp6)) #Use random state 1



## Perform tensor factorization for chosen rank and random state
# cp = parafac(tensor_data, rank=5, normalize_factors=True, random_state=4, init="random")
# print(reconstruction_error(tensor_data, cp))
# cp = parafac(tensor_data, rank=6, normalize_factors=True, random_state=1, init="random")
# cp = parafac(tensor_data, rank=9, normalize_factors=True, init="svd")
# cp = parafac(tensor_data, rank=3, normalize_factors=True, init="svd")
# cp = parafac(tensor_data, rank=12, normalize_factors=True, random_state=4, init="random")



## Plot CPD results
# factorHeatmaps(cp, polyp_type, genes.index.tolist(), cell_types.tolist(), names, 0.1)
# factorHeatmaps(cp, classification, genes.index.tolist(), cell_types.tolist(), names, 0.1)



## Plot gene distributions
# sample_id = pseudobulk_adat.obs["HTAN Specimen ID"].unique()[0]
# cell_type = pseudobulk_adat.obs["Cell_Type"].unique()[0]

# factors = [pd.DataFrame(cp[1][dim],
#                     columns=[f"Comp. {comp+1}" for comp in np.arange(1, len(cp[0])+1)],
#                     index=[samples.tolist(), genes.index.tolist(), cell_types.tolist()][dim])
#             for dim in range(3)]
# factors[1] = factors[1][(factors[1] > 0.02).any(axis=1)]
# gene_subset = factors[1].index.tolist()
# # gene_subset = np.append(gene_subset, ["B2M", "FABP1", "IGKC"])

# geneDistributions(pseudobulk_adat, cell_type, gene_subset)



## Plot mean expression of weird gene
# plot_mean_expression(adat, "IGKC", "SSC")