#!/usr/bin/env python3
"""
Generate HuggingFace dataset for EmeraldBay from adata_clean.h5ad.gz.

Creates the following HF dataset configurations:
  - expression_data: tokenized single-cell expression profiles
  - gene_metadata: gene symbol / ensembl ID / token ID mapping
  - drug_metadata: unique drug entries
  - cell_line_metadata: cell line annotation (CVCL IDs)
  - summary_statistics: growth rate data (renamed from growth_rate)

Usage:
    python make_emeraldbay_hf_dataset.py
"""

import gc
import json
import logging
import math
import os
import datasets
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

logging.basicConfig(
    format="%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
ADATA_PATH = "/nvme-shared/Data/seqrun_26_27_merged/5/qc/generic/adata_clean.h5ad.gz"
GROWTH_RATE_PATH = (
    "/nvme-shared/Data/seqrun_26_27_merged/5/qc/generic/growth_rate_long.parquet"
)
CELL_LINE_MAPPING_PATH = (
    "/home/shreshth/Barotaxis2/vector_diffusion/scripts/data_prep/cell_gen_ft_a97.csv"
)
VOCAB_JSON_PATH = "/tmp/vevo_v2_vocab.json"
OUTPUT_ROOT = "/nvme-shared/Data/EmeraldBay_HF"


# ── Step 1: Build extended vocabulary ───────────────────────────────────────


def build_extended_vocab(adata_var, vocab_json_path):
    """Build an extended vocabulary that includes all EmeraldBay genes.

    Starts from the existing Tahoe-100M vocab (from JSON), keeps all existing
    gene->token mappings intact, and appends new genes found in EmeraldBay.
    Replaces junk padding tokens and re-pads to a multiple of 64.

    Returns:
        extended_vocab: dict mapping gene_ensembl_id -> token_id
        gene_metadata_rows: list of dicts with gene_symbol, ensembl_id, token_id
    """
    with open(vocab_json_path) as f:
        base_vocab = json.load(f)

    # Separate real tokens from junk/special
    special_tokens = {
        k: v for k, v in base_vocab.items() if k.startswith("<") and "junk" not in k
    }
    junk_tokens = {k: v for k, v in base_vocab.items() if "junk" in k}
    gene_tokens = {
        k: v for k, v in base_vocab.items() if not k.startswith("<") and "junk" not in k
    }

    log.info(
        f"Base vocab: {len(special_tokens)} special, {len(gene_tokens)} genes, {len(junk_tokens)} junk",
    )

    # Find genes in EmeraldBay that are NOT in the existing vocab
    eb_genes = list(zip(adata_var.index, adata_var["gene_id"].values))
    existing_ensembl = set(gene_tokens.keys())
    new_genes = [
        (symbol, eid) for symbol, eid in eb_genes if eid not in existing_ensembl
    ]
    log.info(
        f"EmeraldBay genes: {len(eb_genes)}, already in vocab: {len(eb_genes) - len(new_genes)}, new: {len(new_genes)}",
    )

    # Sort junk tokens by token_id to know which IDs to reclaim
    junk_sorted = sorted(junk_tokens.items(), key=lambda x: x[1])
    junk_ids = [v for _, v in junk_sorted]

    # Assign token IDs to new genes:
    #  - First, reclaim junk token IDs
    #  - Then, append after the last used ID
    extended_vocab = dict(base_vocab)  # start with full copy
    # Remove all junk tokens
    for k in junk_tokens:
        del extended_vocab[k]

    next_id = max(base_vocab.values()) + 1  # after last junk token
    available_ids = list(junk_ids)  # reclaimed junk IDs come first

    for symbol, eid in new_genes:
        if available_ids:
            token_id = available_ids.pop(0)
        else:
            token_id = next_id
            next_id += 1
        extended_vocab[eid] = token_id

    # Re-pad to next multiple of 64
    current_size = len(extended_vocab)
    target_size = math.ceil(current_size / 64) * 64
    num_new_junk = target_size - current_size
    for i in range(num_new_junk):
        extended_vocab[f"<junk{i}>"] = next_id
        next_id += 1

    log.info(f"Extended vocab size: {len(extended_vocab)} (padded to {target_size})")

    # Build gene_metadata rows (all genes, not junk/special)
    ensembl_to_symbol = {eid: sym for sym, eid in eb_genes}
    gene_metadata_rows = []
    for k, v in sorted(extended_vocab.items(), key=lambda x: x[1]):
        if k.startswith("<"):
            continue
        symbol = ensembl_to_symbol.get(k, k)
        gene_metadata_rows.append(
            {
                "gene_symbol": symbol,
                "ensembl_id": k,
                "token_id": v,
            },
        )

    return extended_vocab, gene_metadata_rows


# ── Step 2: Generate expression_data ────────────────────────────────────────


def expression_data_generator(adata_path, extended_vocab, cell_line_map):
    """Generator yielding one dict per cell for expression_data config."""
    log.info(f"Loading adata from {adata_path}")
    adata = sc.read_h5ad(adata_path, backed="r")
    log.info(f"Loaded adata: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Map genes to token IDs
    adata.var.reset_index(inplace=True)
    gene_col = "gene_id"
    adata.var["token_id"] = [extended_vocab.get(g, -1) for g in adata.var[gene_col]]
    # Keep ALL genes (should be 100% coverage with extended vocab)
    n_unmapped = (adata.var["token_id"] == -1).sum()
    if n_unmapped > 0:
        log.warning(
            f"{n_unmapped} genes not in extended vocab — this should not happen!",
        )
    adata = adata[:, adata.var["token_id"] >= 0]
    gene_token_ids = np.array(adata.var["token_id"])

    # Load count matrix
    count_matrix = adata.X
    if isinstance(count_matrix, np.ndarray):
        count_matrix = csr_matrix(count_matrix)
    elif isinstance(count_matrix, csc_matrix):
        count_matrix = count_matrix.tocsr()
    elif hasattr(count_matrix, "to_memory"):
        count_matrix = count_matrix.to_memory().tocsr()

    obs = adata.obs.copy()
    obs.reset_index(inplace=True)
    index_col = "BARCODE_SUB_LIB_ID"

    n_cells = count_matrix.shape[0]
    for idx in range(n_cells):
        if idx % 100_000 == 0:
            log.info(f"Processing cell {idx}/{n_cells}")

        row = count_matrix.getrow(idx)
        nonzero_idx = row.indices
        values = row.data

        genes = gene_token_ids[nonzero_idx]

        # Map cell_line from internal c_X to CVCL
        raw_cl = str(obs.iloc[idx]["cell_line"])
        cell_line = cell_line_map.get(raw_cl, raw_cl)

        yield {
            "genes": genes.tolist(),
            "expressions": values.astype(np.float32).tolist(),
            "drug": str(obs.iloc[idx]["drug"]),
            "drugname_drugconc": str(obs.iloc[idx]["drugname_drugconc"]),
            "cell_line": cell_line,
            "sample": str(obs.iloc[idx]["sample"]),
            "BARCODE_SUB_LIB_ID": str(obs.iloc[idx][index_col]),
        }

    del adata, count_matrix
    gc.collect()


# ── Step 3: Build metadata tables ──────────────────────────────────────────


def build_drug_metadata(adata_path):
    """Extract unique drug metadata from adata obs."""
    adata = sc.read_h5ad(adata_path, backed="r")
    obs = adata.obs[["drug", "drugname_drugconc"]].copy()
    obs = obs.drop_duplicates()
    obs = obs.sort_values("drug").reset_index(drop=True)
    records = obs.to_dict("records")
    for r in records:
        r["drug"] = str(r["drug"])
        r["drugname_drugconc"] = str(r["drugname_drugconc"])
    del adata
    gc.collect()
    return records


def build_cell_line_metadata(cell_line_csv_path, adata_path):
    """Build cell_line_metadata from mapping CSV, filtered to EmeraldBay cell lines."""
    cl_df = pd.read_csv(cell_line_csv_path)

    # Get the set of cell lines in EmeraldBay
    adata = sc.read_h5ad(adata_path, backed="r")
    eb_cell_lines = set(adata.obs["cell_line"].unique())
    del adata
    gc.collect()

    # Filter to EmeraldBay cell lines
    cl_df = cl_df[cl_df["Cell_ID_Vevo"].isin(eb_cell_lines)].copy()
    log.info(f"Cell line metadata: {len(cl_df)} lines (filtered from mapping CSV)")

    # Drop internal-only columns
    drop_cols = ["Created By", "Last Modified By"]
    cl_df = cl_df.drop(columns=[c for c in drop_cols if c in cl_df.columns])

    # Fix mixed-type columns: fill NaN strings with ""
    str_cols = cl_df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        cl_df[col] = cl_df[col].fillna("").astype(str)

    return cl_df.to_dict("records")


def build_summary_statistics(growth_rate_path, cell_line_map):
    """Build summary_statistics from growth_rate parquet, mapping cell lines to CVCL."""
    df = pd.read_parquet(growth_rate_path)
    df["cell_line"] = df["cell_line"].map(cell_line_map).fillna(df["cell_line"])
    df["condition"] = df["condition"].astype(str)
    return df.to_dict("records")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load cell line mapping (c_X -> CVCL)
    cl_df = pd.read_csv(CELL_LINE_MAPPING_PATH)
    cell_line_map = dict(zip(cl_df["Cell_ID_Vevo"], cl_df["Cell_ID_Cellosaur"]))
    log.info(f"Loaded cell line mapping: {len(cell_line_map)} entries")

    # ── Extended vocabulary ──────────────────────────────────────────────
    log.info("Building extended vocabulary...")
    adata_tmp = sc.read_h5ad(ADATA_PATH, backed="r")
    extended_vocab, gene_metadata_rows = build_extended_vocab(
        adata_tmp.var,
        VOCAB_JSON_PATH,
    )
    del adata_tmp
    gc.collect()

    # Save extended vocab JSON
    vocab_out_path = os.path.join(OUTPUT_ROOT, "extended_vocab.json")
    with open(vocab_out_path, "w") as f:
        json.dump(extended_vocab, f, indent=2)
    log.info(f"Saved extended vocab to {vocab_out_path}")

    # ── Gene metadata ────────────────────────────────────────────────────
    log.info("Saving gene_metadata...")
    gene_meta_ds = datasets.Dataset.from_list(gene_metadata_rows)
    gene_meta_path = os.path.join(OUTPUT_ROOT, "gene_metadata")
    gene_meta_ds.save_to_disk(gene_meta_path)
    log.info(f"gene_metadata: {len(gene_meta_ds)} rows -> {gene_meta_path}")
    del gene_meta_ds
    gc.collect()

    # ── Drug metadata ────────────────────────────────────────────────────
    log.info("Building drug_metadata...")
    drug_records = build_drug_metadata(ADATA_PATH)
    drug_meta_ds = datasets.Dataset.from_list(drug_records)
    drug_meta_path = os.path.join(OUTPUT_ROOT, "drug_metadata")
    drug_meta_ds.save_to_disk(drug_meta_path)
    log.info(f"drug_metadata: {len(drug_meta_ds)} rows -> {drug_meta_path}")
    del drug_meta_ds
    gc.collect()

    # ── Cell line metadata ───────────────────────────────────────────────
    log.info("Building cell_line_metadata...")
    cl_records = build_cell_line_metadata(CELL_LINE_MAPPING_PATH, ADATA_PATH)
    cl_meta_ds = datasets.Dataset.from_list(cl_records)
    cl_meta_path = os.path.join(OUTPUT_ROOT, "cell_line_metadata")
    cl_meta_ds.save_to_disk(cl_meta_path)
    log.info(f"cell_line_metadata: {len(cl_meta_ds)} rows -> {cl_meta_path}")
    del cl_meta_ds
    gc.collect()

    # ── Summary statistics ───────────────────────────────────────────────
    log.info("Building summary_statistics...")
    ss_records = build_summary_statistics(GROWTH_RATE_PATH, cell_line_map)
    ss_ds = datasets.Dataset.from_list(ss_records)
    ss_path = os.path.join(OUTPUT_ROOT, "summary_statistics")
    ss_ds.save_to_disk(ss_path)
    log.info(f"summary_statistics: {len(ss_ds)} rows -> {ss_path}")
    del ss_ds
    gc.collect()

    # ── Expression data (the big one) ────────────────────────────────────
    log.info("Generating expression_data (this will take a while)...")
    expr_ds = datasets.Dataset.from_generator(
        expression_data_generator,
        gen_kwargs={
            "adata_path": ADATA_PATH,
            "extended_vocab": extended_vocab,
            "cell_line_map": cell_line_map,
        },
        keep_in_memory=False,
    )
    log.info(f"expression_data generated: {len(expr_ds)} rows")

    # Save as single "train" split (no train/valid split)
    train_path = os.path.join(OUTPUT_ROOT, "expression_data", "train")
    os.makedirs(train_path, exist_ok=True)
    expr_ds.save_to_disk(train_path)
    log.info(f"expression_data train: {len(expr_ds)} rows -> {train_path}")

    del expr_ds
    gc.collect()
    log.info("Done! All datasets saved to %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
