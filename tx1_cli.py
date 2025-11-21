# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import OmegaConf as om

# define Typer app
app = typer.Typer(help="Tx1 command line interface.")


# helper function to apply configuration overrides
def _apply_overrides(cfg, overrides):
    if not overrides:
        return cfg
    changes = {}
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(f"Override must be key=value, got: {item}")
        key, val = item.split("=", 1)
        try:
            parsed = om.create(val)
        except Exception:
            parsed = val
        om.update(changes, key, parsed, merge=True)
    return om.merge(cfg, changes)


# root command
@app.callback(invoke_without_command=True)
def root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Tx1 command line interface.")
        typer.echo("Use `tx1 emb` to extract embeddings from an AnnData (.h5ad) file.")
        typer.echo("Run `tx1 emb --help` for detailed options.")


# emb command
@app.command("emb")
def emb(
    f: Optional[Path] = typer.Option(
        None,
        "-f",
        "--config",
        help="Path to YAML config.",
    ),
    set_: Optional[List[str]] = typer.Option(
        None,
        "--set",
        help="Override config values (repeatable), e.g. --set paths.adata_input=data.h5ad",
    ),
    hf_repo: Optional[str] = typer.Option(
        None,
        "--hf-repo",
        help="HF repo id, e.g. tahoebio/Tahoe-x1",
    ),
    model_size: Optional[str] = typer.Option(
        None,
        "--model-size",
        help="Model size: 70m, 1b, or 3b",
    ),
    input: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input .h5ad",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad",
    ),
    cell_label_key: Optional[str] = typer.Option(
        None,
        "--cell-label-key",
        help="Cell label key in adata.obs",
    ),
    gene_id_key: Optional[str] = typer.Option(
        None,
        "--gene-id-key",
        help="Gene ID key in adata.var",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Batch size for inference (default: 512)",
    ),
    seq_len: Optional[int] = typer.Option(
        None,
        "--seq-len",
        help="Sequence length to use during inference (default: 2048)",
    ),
    return_gene_embeddings: bool = typer.Option(
        False,
        "--return-gene-embs",
        help="Also compute gene embeddings",
    ),
):

    # build configuration
    if f:
        cfg = om.load(str(f))
        cfg = _apply_overrides(cfg, set_ or [])
    else:
        if not (
            hf_repo
            and model_size
            and input
            and output
            and cell_label_key
            and gene_id_key
        ):
            raise typer.BadParameter(
                "Provide either -f CONFIG.yaml or flags: --hf-repo, --model-size, --input, --output (and optional others).",
            )
        cfg = om.create(
            {
                "model_name": "tx1",
                "paths": {
                    "hf_repo_id": hf_repo,
                    "hf_model_size": model_size,
                    "adata_input": str(input),
                    "adata_output": str(output),
                },
                "data": {
                    "cell_type_key": cell_label_key,
                    "gene_id_key": gene_id_key,
                },
                "predict": {
                    "batch_size": batch_size or 512,
                    "seq_len_dataset": seq_len if seq_len is not None else 2048,
                    "return_gene_embeddings": bool(return_gene_embeddings),
                },
            },
        )

    # run prediction
    from tahoe_x1.inference import predict_embeddings

    _ = predict_embeddings(cfg)
    typer.echo("Done.")


# main function
def main():
    app()


# entry point
if __name__ == "__main__":
    main()
