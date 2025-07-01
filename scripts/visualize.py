import click
import subprocess
import sys
from pathlib import Path

from src import load_config_from_yaml, AppConfig

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
    Launches Streamlit visualization for the experiment specified in the config file.
    """
    config_path = Path(config_path_str)
    try:
        cfg: AppConfig = load_config_from_yaml(config_path)
    except ValueError as e:
        click.secho(f"Error loading config: {e}", fg="red")
        return

    output_dir = Path(cfg.data.output_path)
    target_dir = output_dir / cfg.model.model_name.replace('/', '_') / f"window_{cfg.experiment.context_window}" / cfg.experiment.target_word
    
    if not target_dir.exists():
        click.secho(f"Error: Output directory does not exist: {target_dir}", fg="red")
        click.secho("Make sure you have run the experiment first.", fg="yellow")
        return
    
    project_root = Path(__file__).parent.parent
    streamlit_app = project_root / "streamlit" / "landscape_viewer.py"
    
    if not streamlit_app.exists():
        click.secho(f"Error: Streamlit app not found: {streamlit_app}", fg="red")
        return
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_app), "--", str(target_dir), cfg.experiment.target_word],
        stderr=subprocess.DEVNULL) # This suppresses Streamlit's stderr output when looking at torch stuff

if __name__ == "__main__":
    main()