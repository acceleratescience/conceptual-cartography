# main.py
import click
from pathlib import Path
import torch

from src.embeddings import ContextEmbedder
from src.config import AppConfig
from src.utils import load_config_from_yaml, load_sentences, save_output

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
    cfg: AppConfig = load_config_from_yaml(config_path)

    click.echo(f"🚀 Starting experiment with configuration from: {config_path}")
    click.echo(f"Model: {cfg.model_configs.model_name}")
    click.echo(f"Target word: '{cfg.experiment_configs.target_word}'")

    # 1. Initialize ContextEmbedder
    click.echo("Initializing ContextEmbedder...")
    embedder = ContextEmbedder(model_name=cfg.model_configs.model_name)
    click.echo(f"ContextEmbedder initialized with device: {embedder.device}")

    # 2. Load sentences
    click.echo(f"Loading sentences from: {cfg.data_configs.sentences_path}...")
    try:
        sentences = load_sentences(cfg.data_configs.sentences_path)
    except FileNotFoundError:
        click.secho(f"Error: Sentences file not found at {cfg.data_configs.sentences_path}", fg="red")
        return
    click.echo(f"Loaded {len(sentences)} sentences.")
    if not sentences:
        click.secho("No sentences loaded. Exiting.", fg="yellow")
        return

    # 3. Run embedding process
    click.echo("Generating embeddings...")
    output = embedder(
        sentences=sentences,
        target_word=cfg.experiment_configs.target_word,
        context_window=cfg.experiment_configs.context_window,
        model_batch_size=cfg.experiment_configs.model_batch_size,
    )

    if output['final_embeddings'].numel() == 0 or output['final_embeddings'].shape == (1, embedder.model.config.hidden_size) and torch.all(output['final_embeddings'] == 0):
        click.secho(f"No embeddings were generated for the target word '{cfg.experiment_configs.target_word}'. Check your data and target word.", fg="yellow")
    else:
        click.echo(f"Generated {output['final_embeddings'].shape[0]} embeddings of dimension {output['final_embeddings'].shape[1]}.")

        # 4. Save results
        output_dir = Path(cfg.data_configs.output_path)
        # Add model name and target word to output path for better organization
        output_subdir = output_dir / cfg.model_configs.model_name.replace('/', '_') / cfg.experiment_configs.target_word
        
        click.echo(f"Saving results to: {output_subdir}...")
        save_output(str(output_subdir), output)
        click.secho("✅ Experiment finished successfully!", fg="green")

if __name__ == '__main__':
    main()