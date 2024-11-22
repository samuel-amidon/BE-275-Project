import numpy as np
import pandas as pd
import scanpy as sc
from pseudobulkFuncs import pseudobulk
from tensorFactorizationFuncs import tensor_restructure, tensor_factorization


adat = sc.read_h5ad("/Users/samuelamidon/Desktop/GitRepos/VUMC_HTAN_DIS_EPI_V2.h5ad")


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




# x = []

# x = np.append(x, np.array([[1,2,3],[4,5,6],[7,8,9]]))
# x = np.append(x, np.array([[1,2,3],[4,5,6],[7,8,9]]))
# x = np.append(x, np.array([[1,2,3],[4,5,6],[7,8,9]]))

# print(x)

# x1 = np.array(x)

# print(x1)


# x = np.array([[1,2,3],[1,2,6],[1,2,9],[1,2,12]])
# y = x / x.mean(axis=0)
# print(x.mean(axis=0))
# print(y)