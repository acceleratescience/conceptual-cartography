import click
import subprocess
import sys
from pathlib import Path

@click.command()
@click.option(
    '--path',
    'path_str',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="Path to the output directory containing metrics and landscapes."
)
def main(path_str: str):
    """
    Launches Streamlit visualization for the specified output directory.
    """
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/visualize.py", "--", path_str],
        stderr=subprocess.DEVNULL) # This suppresses Streamlit's stderr output when looking at torch stuff.

if __name__ == "__main__":
    main()