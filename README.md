<!-- PROJECT SHIELDS -->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url] -->
[![MIT License](https://img.shields.io/badge/License-MIT%20v3-green.svg)](https://opensource.org/licenses/)
[![Issues](https://img.shields.io/github/issues-raw/acceleratescience/conceptual-cartography.svg?maxAge=25000)](https://github.com/acceleratescience/conceptual-cartography/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/acceleratescience/conceptual-cartography.svg?style=flat)](https://github.com/acceleratescience/conceptual-cartography/graphs/contributors)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/acceleratescience/conceptual-cartography.svg?style=flat)](https://github.com/acceleratescience/conceptual-cartography/pulls)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
<br>
[![GitHub stars](https://img.shields.io/github/stars/acceleratescience/conceptual-cartography.svg?style=social&label=Star)]()
[![GitHub watchers](https://img.shields.io/github/watchers/acceleratescience/conceptual-cartography.svg?style=social&label=Watch)]()
[![GitHub forks](https://img.shields.io/github/forks/acceleratescience/conceptual-cartography.svg?style=social&label=Fork)]()
[![GitHub followers](https://img.shields.io/github/followers/acceleratescience.svg?style=social&label=Follow)](https://github.com/acceleratescience)
[![Twitter Follow](https://img.shields.io/twitter/follow/AccelerateSci.svg?style=social)](https://twitter.com/AccelerateSci)
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

# <img src="assets/bert.png" alt="BERT" width="40" style="vertical-align: middle;"/> Conceptual Landscaping with BERT and Friends

Welcome to the Conceptual Landscaping with BERT and Friends repository! This project is designed to help you explore and visualize the conceptual landscape of a given text using advanced language models like BERT, RoBERTa, and DistilBERT. The goal is to provide a user-friendly interface for analyzing and understanding the relationships between different concepts in your text.

## Installation
To run this code, we first recommend forking the repo so you have your own version to play with. Clone the repo

```bash
git clone https://github.com/acceleratescience/conceptual-cartography.git
cd conceptual-engineering
```

The `conceptual-engineering` repo comes with a `setup.sh` file that will handle most of the installation and environment management for you. This means that if you want to run this software on a remote cloud server, then it's as simple as spinning up a CPU or GPU instance with some base Linux such as Ubuntu, cloning the repo, and running the setup.

We first recommend installing python 3.12. Instructions for Linux are below, and other operating systems such as MacOS are easy to find.

```bash
sudo apt-get update
sudo apt-get install python3.12
```
Now when the setup file:
```bash
source ./setup.sh
```
After a bunch of install infomation, you should see something like the following:

```bash
Installing the current project: conceptual-cartography (0.1.0)
✓ Poetry environment created successfully

=== Setup Status ===

✓ Setup Complete with regular dependencies! 🎉
✓ To activate the virtual environment, run: poetry shell
✓ Or use: source .venv/bin/activate
✓ To run commands in the environment: poetry run <command>

```
Do what it says and activate your virtual environment!

You can also run the setup with development dependencies by adding the flag `--dev`

## Installation of PyTorch
Due to the nature of PyTorch installations across different hardware, we have left the installation of PyTorch to the user. For installation instructions, please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/). The majority of the development for this repo was done using ROCm...not CUDA.

If using an AMD GPU, you can install PyTorch with ROCm support:

```bash
 pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
```

Then run the following:

```bash
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # adjust version for your GPU
```
To make it permanent:
```bash
echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc  # adjust version for your GPU
```
then restart your terminal or run `source ~/.bashrc`. Don't forget to reactivate your virtual environment if you are using one.

### CPU only
For cpu only PyTorch, you can run
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data Format Requirements

**Important**: Your input text data must be formatted with **one sentence per line**. The system reads text files line-by-line, where each non-empty line is treated as a separate sentence for analysis.

### Example format:
```
I need to go to the bank today to deposit this cheque.
She decided to open a new savings account at a different bank for a better interest rate.
The bank approved their mortgage application after reviewing their financial history.
We decided to have our picnic on the grassy bank of the river.
The children loved skipping stones from the river bank into the water.
```

### ❌ Incorrect format:
```
I need to go to the bank today. She decided to open a savings account. The bank approved their mortgage application.
```

## Running an experiment.
The file directory should look something like:

```
.
├── configs
│   └── bank-test-metrics.yaml
├── data
│   ├── testing_data
│   │   └── bank_test.txt
├── scripts
├── src
...etc

```

The important thing here is that we have some testing data `./data/testing_data/bank_test.txt` and a testing config file `./configs/bank-test-metrics.yaml`

### Configs
This config file should look something like this
```yaml
model:
  model_name: 'sentence-transformers/all-MiniLM-L6-v2'
data:
  sentences_path: 'data/testing_data/bank_test.txt'
  output_path: 'output'
experiment:
  model_batch_size: 32
  context_window: None
  target_word: 'bank'
metrics:
  anisotropy_correction: False
  layers: 'all'
  metrics: ['similarity_matrix', 'mev', 'inter_similarity', 'intra_similarity', 'average_similarity', 'similarity_std']
landscapes:
  pca_min: 2
  pca_max: 5
  pca_step: 1
  cluster_min: 2
  cluster_max: 5
  cluster_step: 1
  generate_all: True
  save_optimization: True
```
### Running experiments on a corpus
To run this file, simply run in the command line
```bash
poetry run experiment --config 'configs/bank-test-metrics.yaml'
```
When running for the first time, you will see the model being downloaded. The corpus examples will be run through the model, and the landscapes will be saved and generated.

You should now have a new directory:
```bash
├── output
│   └── sentence-transformers_all-MiniLM-L6-v2
│       └── window_None
│           └── bank
│               ├── contexts.txt
│               ├── final_embeddings.pt
│               ├── hidden_embeddings.pt
│               ├── indices.txt
│               ├── landscapes
│               │   ├── landscape_layer-0.pt
│               │   ├── landscape_layer-1.pt
│               │   ├── landscape_layer-2.pt
│               │   ├── landscape_layer-3.pt
│               │   ├── landscape_layer-4.pt
│               │   └── landscape_layer-5.pt
│               └── metrics
│                   ├── metrics_layer-0.pt
│                   ├── metrics_layer-1.pt
│                   ├── metrics_layer-2.pt
│                   ├── metrics_layer-3.pt
│                   ├── metrics_layer-4.pt
│                   └── metrics_layer-5.pt
```
Each landscape `.pt` file is a `Landscape` object containing the information required to visualize the conceptual landscapes:
```python
class Landscape:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    X_pca: np.ndarray
    consensus_labels: np.ndarray
    ari_scores: list
    pca_components: int = None
    cluster_count: int = None
    covariance_type: str = None
```

### Visualizing the experiment
Visualizing a completed experiment is straight forward:
```bash
poetry run visualize --config 'configs/bank-test-metrics.yaml'
```
This will then show something like
```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.17.0.6:8501
  External URL: http://213.173.105.105:8501
```

Head to the url to open the app and click through the layers to explore the different clusters!

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/acceleratescience/conceptual-cartography.svg?style=for-the-badge
[contributors-url]: https://github.com/acceleratescience/conceptual-cartography/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/acceleratescience/conceptual-cartography.svg?style=for-the-badge
[forks-url]: https://github.com/acceleratescience/conceptual-cartography/network/members
[stars-shield]: https://img.shields.io/github/stars/acceleratescience/conceptual-cartography.svg?style=for-the-badge
[stars-url]: https://github.com/acceleratescience/conceptual-cartography/stargazers
[issues-shield]: https://img.shields.io/github/issues/acceleratescience/conceptual-cartography.svg?style=for-the-badge
[issues-url]: https://github.com/acceleratescience/conceptual-cartography/issues
[license-shield]: https://img.shields.io/github/license/acceleratescience/conceptual-cartography.svg?style=for-the-badge
[license-url]: https://github.com/acceleratescience/conceptual-cartography/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/company/accelerate-programme-for-scientific-discovery/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 