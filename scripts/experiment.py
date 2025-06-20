# main.py
import click
from pathlib import Path
import torch
from tqdm import tqdm

from src import (
    ContextEmbedder,
    AppConfig,
    load_config_from_yaml,
    load_sentences,
    save_output,
    save_metrics,
    save_landscape,
    EmbeddingMetrics,
    get_landscape,
    optimize_clustering,
)

@click.command()
@click.option(
    '--config',
    'config_path_str',
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the YAML configuration file."
)
def main(config_path_str: str):
    """
    Runs an embedding experiment based on the provided YAML configuration.
    """
    config_path = Path(config_path_str)
    try:
        cfg: AppConfig = load_config_from_yaml(config_path)
    except ValueError as e:
        return

    # Check if metrics are provided and set the flag
    cfg.metrics.metrics_provided = bool(cfg.metrics.metrics)
    cfg.landscapes.landscapes_provided = bool(cfg.landscapes.pca_min)

    click.echo(f"🚀 Starting experiment with configuration from: {config_path}")
    click.echo(f"Model: {cfg.model.model_name}")
    click.echo(f"Target word: '{cfg.experiment.target_word}'")

    # 1. Initialize ContextEmbedder
    click.echo("Initializing ContextEmbedder...")
    embedder = ContextEmbedder(model_name=cfg.model.model_name)
    click.echo(f"ContextEmbedder initialized with device: {embedder.device}")

    # 2. Load sentences
    click.echo(f"Loading sentences from: {cfg.data.sentences_path}...")
    try:
        sentences = load_sentences(cfg.data.sentences_path)
    except FileNotFoundError:
        click.secho(f"Error: Sentences file not found at {cfg.data.sentences_path}", fg="red")
        return
    click.echo(f"Loaded {len(sentences)} sentences.")
    if not sentences:
        click.secho("No sentences loaded. Exiting.", fg="yellow")
        return

    # 3. Run embedding process
    click.echo("Generating embeddings...")
    output = embedder(
        sentences=sentences,
        target_word=cfg.experiment.target_word,
        context_window=cfg.experiment.context_window,
        model_batch_size=cfg.experiment.model_batch_size,
    )

    if output['final_embeddings'].numel() == 0 or output['final_embeddings'].shape == (1, embedder.model.config.hidden_size) and torch.all(output['final_embeddings'] == 0):
        click.secho(f"No embeddings were generated for the target word '{cfg.experiment.target_word}'. Check your data and target word.", fg="yellow")
    else:
        click.echo(f"Generated {output['final_embeddings'].shape[0]} embeddings of dimension {output['final_embeddings'].shape[1]}.")

        # 4. Save results
        output_dir = Path(cfg.data.output_path)
        output_subdir = output_dir / cfg.model.model_name.replace('/', '_') / f"window_{cfg.experiment.context_window}" / cfg.experiment.target_word 
        
        click.echo(f"Saving embedding results to: {output_subdir}...")
        save_output(str(output_subdir), output)
        click.secho("✅ Embeddings generated and saved successfully!", fg="green")

    # Now we can check to see if the metrics and the landscapes are provided. If the landscapes are provided,
    # that means we will also be generating labels. If this is the case, the metrics need to be calculated after the landscapes.
    # 

    labels = None

    # 5. Generate landscapes if configured
    if cfg.landscapes.landscapes_provided:
        layers = cfg.metrics.layers
        labels = []
        click.echo("Generating landscapes...")
        
        if layers == 'final':
            embeddings = output['final_embeddings']
        else:
            for layer in tqdm(layers, desc="Generating landscapes..."):
                embeddings = output['hidden_embeddings'][:, layer, :]
                best_params, df = optimize_clustering(
                    embeddings,
                    pca_components_range=range(cfg.landscapes.pca_min, cfg.landscapes.pca_max + 1, cfg.landscapes.pca_step),
                    n_clusters_range=range(cfg.landscapes.cluster_min, cfg.landscapes.cluster_max + 1, cfg.landscapes.cluster_step),
                )

                landscape = get_landscape(
                    embeddings,
                    best_params['best_params']
                )

                if cfg.landscapes.generate_all:
                    output_subdir = output_dir / cfg.model.model_name.replace('/', '_') / f"window_{cfg.experiment.context_window}" / cfg.experiment.target_word / "landscapes"
                    save_landscape(str(output_subdir), layer, landscape)

                labels.append(landscape.consensus_labels)
        click.secho("✅ Landscapes calculated and saved successfully!", fg="green")

    # 6. Calculate metrics
    if cfg.metrics.metrics_provided:
        click.echo("Calculating metrics...")
        
        if layers == 'final':
            layer = 'final'
            final_layer = output['final_embeddings']
            metrics = EmbeddingMetrics(embeddings=final_layer, labels=None)
            metrics_dict = metrics.get_metrics(
                corrected=cfg.metrics.anisotropy_correction,
                include=cfg.metrics.metrics
            )
            output_subdir = output_dir / cfg.model.model_name.replace('/', '_') / f"window_{cfg.experiment.context_window}" / cfg.experiment.target_word / "metrics"
            save_metrics(str(output_subdir), layer, metrics_dict)
        else:
            if layers == 'all':
                layers = range(output['hidden_embeddings'].shape[1])
            for i, layer in tqdm(enumerate(layers), desc="Processing layers..."):
                hidden_layer = output['hidden_embeddings'][:, layer, :]
                metrics = EmbeddingMetrics(embeddings=hidden_layer, labels=labels[i] if labels else None)
                metrics_dict = metrics.get_metrics(
                    corrected=cfg.metrics.anisotropy_correction,
                    include=cfg.metrics.metrics
                )
                output_subdir = output_dir / cfg.model.model_name.replace('/', '_') / f"window_{cfg.experiment.context_window}" / cfg.experiment.target_word / "metrics"
                save_metrics(str(output_subdir), layer, metrics_dict)
        click.secho("✅ Metrics calculated and saved successfully!", fg="green")

    

if __name__ == '__main__':
    main()