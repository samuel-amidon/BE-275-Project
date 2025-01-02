import scanpy as sc

from tensorly.decomposition import parafac

from project.pseudobulkFuncs import pseudobulk
from project.tensorFactorizationFuncs import tensor_restructure, reconstruction_error
from project.figures.analysis_comparison import analysisComparison
from project.figures.reconstruction_error_plot import errorPlot
from project.figures.factor_plots import factorHeatmaps
from project.figures.plot_raw_expressions import meanExpression


# Read raw data
adat = sc.read_h5ad("/Users/samuelamidon/Desktop/BE275 Project Supp/VUMC_HTAN_DIS_EPI_V2.h5ad")

# Process raw data
pseudobulk_adat = pseudobulk(adat, min_expression=0.01)

# Filter genes
genes_to_remove = ["LINC00486","MT-ATP6","MT-CO1","MT-CO2","MT-CO3","MT-CYB","MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-RNR1","MT-RNR2","RP5-857K21.4"]
pseudobulk_adat_filt = pseudobulk_adat[:, ~pseudobulk_adat.var_names.isin(genes_to_remove)]

# Restructure data
tensor_data = tensor_restructure(pseudobulk_adat_filt)

# Get metadata
samples = pseudobulk_adat_filt.obs["HTAN Specimen ID"].unique()
genes = pseudobulk_adat_filt.var
cell_types = pseudobulk_adat_filt.obs["Cell_Type"].unique()
classification = [
        pseudobulk_adat_filt.obs["Sample_Classification"][pseudobulk_adat_filt.obs["HTAN Specimen ID"] == sample].iloc[0]
        for sample in samples
    ]
polyp_type = [
        pseudobulk_adat_filt.obs["Polyp_Type"][pseudobulk_adat_filt.obs["HTAN Specimen ID"] == sample].iloc[0]
        for sample in samples
    ]

# Perform tensor factorization for all ranks with multiple runs
cp_ls = [[], [], [], []]
err_ls = [[], [], [], []]
rank = 9
for i in range(4):
    for j in range(rank):
        cp_ls[i].append(parafac(tensor_data, rank=j+1, normalize_factors=True, init="random"))
        err_ls[i].append(reconstruction_error(tensor_data, cp_ls[i][j]))

# Make reconstruction error figure
errorPlot(err_ls) # Overlay all runs to look at run-to-run variability
errorPlot(err_ls[0]) # Look at one run to see how error changes with rank

# Make analysis comparison figure with PCA
analysisComparison(err_ls, tensor_data)

# Find random state that gives lowest error
for i in range(10):
    cp = parafac(tensor_data, rank=5, normalize_factors=True, random_state=int(i), init="random")
    print("random state:", i, "error:", reconstruction_error(tensor_data, cp))

# Perform tensor factorization for chosen rank
rank = 5
cp = parafac(
    tensor_data, 
    rank=rank, 
    normalize_factors=True,
    random_state=4, # Use random state found in previous section
    init="random"
)

# Make factor heatmap figure
factorHeatmaps(
    cp, 
    lab1=classification, 
    lab2=genes.index.tolist(), 
    lab3=cell_types.tolist(), 
    dim_names=["Samples", "Genes", "Cell Types"], 
    gene_val_lim=0.1
)

# Make mean expression figure
meanExpression(adat, "IGKC", "SSC")