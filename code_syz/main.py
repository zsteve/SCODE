import scipy as sp
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import cellrank as cr
import seaborn as sb
import statot
import pegasus as pg

dir="data/"
t = pd.read_csv(dir + "time.txt", sep = "\t", index_col = 0, header = None)
genes = pd.read_csv(dir + "tf.txt", sep = "\t", index_col = 0, header = None)
df = pd.read_csv(dir + "exp.txt", sep = "\t", header = None, dtype = np.float64).T
df.set_index(t.index, inplace = True)

adata = ad.AnnData(df)
adata.obs['t'] = np.array(t.iloc[:, 0])
adata.var.set_index(genes.index, inplace = True)
# adata.obs.columns = pd.Index(["pseudotime", "day"])
sc.pp.highly_variable_genes(adata, n_top_genes = 25)

sc.tl.pca(adata)
sc.pp.neighbors(adata, 25)
sc.tl.umap(adata)
sc.tl.tsne(adata)

import pegasusio as io
mmdata = io.MultimodalData(adata)
pg.pca(mmdata, n_components = adata_tmp.shape[1], features = None)
pg.neighbors(mmdata, K = 50)
pg.diffmap(mmdata)
pg.fle(mmdata, K = 25, is3d = False, rep = "diffmap")
adata.obsm["X_fle"] = mmdata.obsm["X_fle"]
plt.scatter(adata.obsm["X_fle"][:, 0], adata.obsm["X_fle"][:, 1], c = adata.obs.t)
plt.colorbar()
plt.show()

plt.imshow(adata.X[np.argsort(adata.obs.t), :], interpolation = "none")
plt.axis("auto")
plt.show()

k = cr.tl.kernels.PseudotimeKernel(adata, time_key = "t").compute_transition_matrix() + 0.25*cr.tl.kernels.ConnectivityKernel(adata).compute_transition_matrix()
adata.obsm["P_dpt"] = k.transition_matrix.todense()
x_ord = np.argsort(adata.obs.t)

# construct cost matrix along discrete manifold 
G_sp = adata.uns['neighbors']['distances']
adata.obsm["C"] = sp.sparse.csgraph.floyd_warshall(G_sp, directed = False)**2
# Stationary OT
sink_idx = (adata.obs.t >= np.quantile(adata.obs.t, 0.9)) & (adata.obs.t < np.quantile(adata.obs.t, 1.0))
R = np.zeros(adata.shape[0])
R[sink_idx] = -25/sum(sink_idx)
R[~sink_idx] = -R.sum()/sum(~sink_idx)
gamma, mu, nu = statot.inference.statot(adata.obsm["X_pca"], g = np.exp(R), dt = 1, C = adata.obsm["C"], eps = 1.0*adata.obsm["C"].mean(), method = "quad")
adata.obsm["P_statot"] = statot.inference.row_normalise(gamma)

P_key = "statot"
plt.imshow(np.array(adata.obsm["P_%s" % P_key])[x_ord, :][:, x_ord], vmax = np.quantile(adata.obsm["P_%s" % P_key], 0.995))
plt.show()

u, v= sp.linalg.eig(np.array(adata.obsm["P_%s" % P_key]).T)
p = np.real(v[:, 0])
p = p/p.sum()
plt.scatter(adata.obsm["X_pca"][:, 0], adata.obsm["X_pca"][:, 1], c = p, vmax = np.quantile(p, 0.995))
plt.colorbar()
plt.show()

dir = "."
# adata = adata[:, np.argsort(adata.var.dispersions_norm)[::-1]]
# write numpy files
# adata = adata[:, adata.var.highly_variable]
np.save(dir + "/X.npy", adata.X)
np.save(dir + "/genes.npy", np.array(adata.var.index, dtype = str))
pd.DataFrame(adata.var.index).to_csv(dir + "/genes.txt")
np.save(dir + "/X_pca.npy", adata.obsm["X_pca"])
np.save(dir + "/X_umap.npy", adata.obsm["X_tsne"])
np.save(dir + "/X_fle.npy", adata.obsm["X_fle"])
# save transition matrices
for k in [x for x in adata.obsm.keys() if "P_" in x]:
    print("Writing transition matrix %s..." % k)
    try:
        np.save(dir + "/%s.npy" % k, np.array(adata.obsm[k].todense()))
    except:
        np.save(dir + "/%s.npy" % k, np.array(adata.obsm[k]))
np.save(dir + "/C.npy", adata.obsm["C"])
np.save(dir + "/dpt.npy", adata.obs["t"].to_numpy())
# np.save(dir + "/J.npy", adata.obsm["J"].to_numpy().reshape(-1, adata.X.shape[1], adata.X.shape[1]))

J_df = pd.read_csv("data/reference_TFTF_network.txt", sep = "\t", header = None)
J_df.columns = pd.Index(["g1", "g2", "x", "y"])
J = pd.DataFrame(np.zeros((adata.shape[1], adata.shape[1])))
J.index = genes.index
J.columns = genes.index
for x in J_df.iloc[:, 0:2].iterrows():
    J.loc[x[1][0], x[1][1]] += 1
np.save(dir + "/J.npy", J.to_numpy())

sb.clustermap(J, row_cluster = False, col_cluster = False)
plt.show()
