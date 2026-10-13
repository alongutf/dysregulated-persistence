"""
UMAP of the concatenated Dis-Arrest (SHX) and Reg-Arrest datasets (Figure 1 G-H).

Cells are the ones used throughout the paper: the barcodes listed in
data_for_paper/<sample>_filtered.csv.  Their raw counts (all genes) are taken
from the per-sample UNFILTERED cell x gene matrices, which are NOT part of this
repository (see "Input data" below).

Pipeline
--------
  1. load raw counts for the paper barcodes of each sample
  2. concatenate samples (inner join on genes), batch labels reg3/dis1/dis2/reg1/reg2
  3. remove 16s rRNA and reporter genes (GENES_TO_REMOVE), filter_genes(min_cells=3)
  4. normalize_total(target_sum=1e4) -- the same target for every sample -- then log1p
  5. top 2000 highly variable genes, scale, PCA (50 comps)
  6. kNN graph (40 neighbours, 40 PCs) -> UMAP (min_dist=0.3) -> Leiden (res=0.4)
  7. Wilcoxon marker genes per Leiden cluster

Genes are removed BEFORE normalisation on purpose: 16s rRNA is ~92% of all
counts, so normalising first would divide every cell by its rRNA content.

Input data (not included in the repository)
-------------------------------------------
Five CSV files, one per sample, cells as rows and genes as columns (first
column = cell barcode, header row = gene names, values = raw UMI counts):

    sample_2b_unfiltered.csv    Reg-Arrest           -> batch 'reg3'
    sample_13a_unfiltered.csv   Dis-Arrest (SHX)     -> batch 'dis1'
    sample_15a_unfiltered.csv   Dis-Arrest (SHX)     -> batch 'dis2'
    sample_13b_unfiltered.csv   Reg-Arrest (casp)    -> batch 'reg1'
    sample_15b_unfiltered.csv   Reg-Arrest (casp)    -> batch 'reg2'

They are the cell x gene count matrices produced from the probe-count CSVs by
scripts/analysis_notebook.ipynb (src.data_functions.get_annotated_data ->
AnnMat.to_csv), before any cell or gene filtering.  Every barcode in
data_for_paper/<sample>_filtered.csv must be present in the matching file.

By default the script looks for them in <repo>/data/ (git-ignored, the same
folder analysis_notebook.ipynb writes to).  To use another folder, either edit
UNFILTERED_DIR below or pass it on the command line:

    python umap_shx.py --unfiltered-dir /path/to/unfiltered_matrices

Outputs (timestamped)
---------------------
  scanpy/adata_shx_<ts>.h5ad              X = scaled, layers: counts, lognorm
  scanpy/umap_coordinates_shx_<ts>.csv    UMAP_1, UMAP_2, batch, cluster
  scanpy/umap_shx_<ts>.svg / .png         two panels: Leiden clusters, sample
  scanpy/marker_genes_shx_<ts>.xlsx       one sheet per cluster
  scripts/logs/umap_shx_<ts>.json         every parameter + per-sample counts
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# >>> SET THIS to the folder holding the five <sample>_unfiltered.csv files <<<
# (or pass --unfiltered-dir).  These files are not distributed with the repo;
# see "Input data" in the module docstring.
UNFILTERED_DIR = os.path.join(REPO, 'data')

PAPER_DIR = os.path.join(REPO, 'data_for_paper')  # source of the cell barcodes
OUT_DIR = os.path.join(REPO, 'scanpy')
LOG_DIR = os.path.join(REPO, 'scripts', 'logs')

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
# sample file prefix -> batch label (dis = Dis-Arrest/SHX, reg = Reg-Arrest)
SAMPLES = [
    ('sample_2b',  'reg3'),
    ('sample_13a', 'dis1'),
    ('sample_15a', 'dis2'),
    ('sample_13b', 'reg1'),
    ('sample_15b', 'reg2'),
]

GENES_TO_REMOVE = ['16s_mature', '16s_unprocessed', 'LELOBEKK', 'kanR', 'mCherry']
MIN_CELLS = 3            # filter_genes: drop genes detected in fewer cells
TARGET_SUM = 1e4         # normalize_total target, identical for every sample
N_TOP_GENES = 2000       # highly variable genes kept for the embedding
N_COMPS = 50             # PCA components
N_NEIGHBORS = 40         # kNN graph size for UMAP / Leiden
N_PCS = 40               # PCs used to build the kNN graph
MIN_DIST = 0.3           # UMAP min_dist
RANDOM_STATE = 0         # UMAP seed
LEIDEN_RESOLUTION = 0.4  # Leiden resolution

POINT_SIZE = 4           # figure only
POINT_ALPHA = 0.35

TS = datetime.now().strftime('%Y%m%d_%H%M%S')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--unfiltered-dir', default=UNFILTERED_DIR,
                        help='folder containing <sample>_unfiltered.csv for all five '
                             f'samples (default: {UNFILTERED_DIR})')
    return parser.parse_args()


def check_inputs(unfiltered_dir):
    """Fail early, with instructions, if any unfiltered matrix is missing."""
    missing = [f'{s}_unfiltered.csv' for s, _ in SAMPLES
               if not os.path.isfile(os.path.join(unfiltered_dir, f'{s}_unfiltered.csv'))]
    if missing:
        raise FileNotFoundError(
            f'Unfiltered count matrices not found in:\n    {unfiltered_dir}\n'
            f'Missing: {", ".join(missing)}\n\n'
            'These files are not included in the repository. Generate them from the '
            'probe-count CSVs with scripts/analysis_notebook.ipynb, then either place '
            'them in that folder, set UNFILTERED_DIR at the top of this script, or run:\n'
            '    python umap_shx.py --unfiltered-dir /path/to/unfiltered_matrices')


def load_sample(sample, unfiltered_dir, log):
    """Raw counts (all genes) for the paper barcodes of one sample."""
    barcodes = pd.read_csv(os.path.join(PAPER_DIR, f'{sample}_filtered.csv'),
                           index_col=0, usecols=[0]).index
    mat = pd.read_csv(os.path.join(unfiltered_dir, f'{sample}_unfiltered.csv'),
                      index_col=0)
    absent = barcodes.difference(mat.index)
    if len(absent):
        raise ValueError(f'{sample}: {len(absent)} paper barcodes absent from the '
                         f'unfiltered matrix (first few: {list(absent[:5])})')
    mat = mat.loc[barcodes]

    log[sample] = {
        'n_cells': int(mat.shape[0]),
        'n_genes_in': int(mat.shape[1]),
        'median_total_counts': float(mat.sum(1).median()),
    }
    print(f'{sample}: {mat.shape[0]} cells x {mat.shape[1]} genes')

    adata = sc.AnnData(mat.astype('float32'))
    adata.var_names = mat.columns.astype(str)
    adata.obs_names = mat.index.astype(str)
    return adata


def main():
    args = parse_args()
    unfiltered_dir = os.path.abspath(args.unfiltered_dir)
    check_inputs(unfiltered_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log = {'timestamp': TS,
           'script': os.path.basename(__file__),
           'barcode_source': PAPER_DIR,
           'matrix_source': unfiltered_dir,
           'scanpy_version': sc.__version__,
           'params': {
               'order': 'remove genes -> filter_genes -> normalize_total -> log1p',
               'target_sum': TARGET_SUM,
               'genes_to_remove': GENES_TO_REMOVE,
               'filter_genes_min_cells': MIN_CELLS,
               'n_top_genes': N_TOP_GENES,
               'pca_n_comps': N_COMPS, 'pca_svd_solver': 'arpack',
               'n_neighbors': N_NEIGHBORS, 'n_pcs': N_PCS,
               'umap_min_dist': MIN_DIST, 'random_state': RANDOM_STATE,
               'leiden_resolution': LEIDEN_RESOLUTION,
               'rank_genes_method': 'wilcoxon', 'rank_genes_layer': 'counts',
           },
           'samples': {}}

    adatas = [load_sample(s, unfiltered_dir, log['samples']) for s, _ in SAMPLES]
    adata = sc.concat(adatas, label='batch', join='inner', fill_value=0,
                      keys=[k for _, k in SAMPLES], index_unique='-')

    # gene removal BEFORE normalisation, so the normalisation total excludes 16s rRNA
    adata = adata[:, ~adata.var_names.isin(GENES_TO_REMOVE)]
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    adata.layers['counts'] = adata.X.copy()   # raw counts, for marker testing
    print(f'{adata.n_obs} cells x {adata.n_vars} genes after gene removal')

    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)
    adata.layers['lognorm'] = adata.X.copy()  # normalised + log1p, all genes
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, subset=True)
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=N_COMPS)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.umap(adata, min_dist=MIN_DIST, random_state=RANDOM_STATE)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION)
    # marker genes over all genes, tested on raw counts
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon',
                            use_raw=False, layer='counts')

    log['n_cells_total'] = int(adata.n_obs)
    log['n_hvg'] = int(adata.n_vars)
    log['cells_per_batch'] = {k: int(v) for k, v in
                              adata.obs['batch'].value_counts().items()}
    log['cells_per_cluster'] = {k: int(v) for k, v in
                                adata.obs['leiden'].value_counts().items()}

    # ---- outputs ---------------------------------------------------------
    h5ad_path = os.path.join(OUT_DIR, f'adata_shx_{TS}.h5ad')
    adata.write_h5ad(h5ad_path)

    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP_1', 'UMAP_2'],
                           index=adata.obs_names)
    umap_df['batch'] = adata.obs['batch'].values
    umap_df['cluster'] = adata.obs['leiden'].values
    coord_path = os.path.join(OUT_DIR, f'umap_coordinates_shx_{TS}.csv')
    umap_df.to_csv(coord_path)

    # shuffled draw order so no single sample is painted over the others
    shuffled = adata[np.random.default_rng(0).permutation(adata.n_obs)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc.pl.umap(shuffled, color='leiden', ax=axes[0], show=False, frameon=False,
               title='Leiden clusters', size=POINT_SIZE, alpha=POINT_ALPHA)
    sc.pl.umap(shuffled, color='batch', ax=axes[1], show=False, frameon=False,
               title='Sample', size=POINT_SIZE, alpha=POINT_ALPHA)
    fig.tight_layout()
    svg_path = os.path.join(OUT_DIR, f'umap_shx_{TS}.svg')
    png_path = os.path.join(OUT_DIR, f'umap_shx_{TS}.png')
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    markers_path = os.path.join(OUT_DIR, f'marker_genes_shx_{TS}.xlsx')
    with pd.ExcelWriter(markers_path) as writer:
        for cl in adata.obs['leiden'].cat.categories:
            sc.get.rank_genes_groups_df(adata, group=cl).to_excel(
                writer, sheet_name=f'cluster_{cl}', index=False)

    log['outputs'] = {'adata': h5ad_path, 'coordinates': coord_path,
                      'figure_svg': svg_path, 'figure_png': png_path,
                      'markers': markers_path}
    log_path = os.path.join(LOG_DIR, f'umap_shx_{TS}.json')
    with open(log_path, 'w') as fh:
        json.dump(log, fh, indent=2)

    print(f'\n{adata.n_obs} cells, {adata.n_vars} HVGs, '
          f'{len(adata.obs["leiden"].cat.categories)} clusters')
    for name, path in log['outputs'].items():
        print(f'{name:12s}-> {path}')
    print(f'{"log":12s}-> {log_path}')


if __name__ == '__main__':
    main()
