import numpy as np
import pandas as pd
import scanpy as sc
from pseudobulkFuncs import pseudobulk
from tensorFactorizationFuncs import tensor_restructure, tensor_factorization

# Notes
#  adat.raw.X is the same as adat.X


adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/VUMC_HTAN_DIS_EPI_V2.h5ad")
# adat = adat.raw

# print(sum(adat.X[0]))
# print(sum(adat.raw[0]))


# print(adat.raw)
# print(adat.raw.X)

# print(adat.X.mean(axis=0))

# obs = adat.obs.columns

# for i in range(len(obs)):
#     print(obs[i])

# print(adat.obs["HTAN Specimen ID"].unique())
# print(adat.obs["Cell_Type"].unique())
# print(adat)



pseudobulk_adat = pseudobulk(adat)

print(pseudobulk_adat)
print(pseudobulk_adat.X)
print(pseudobulk_adat.obs)
print(pseudobulk_adat.var)

tensor_data = tensor_restructure(pseudobulk_adat)

factors = tensor_factorization(tensor_data, rank=10)