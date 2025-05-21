Conceptual Landscaping with BERT and Friends
========================

Welcome to the Conceptual Landscaping with BERT and Friends repository! This project is designed to help you explore and visualize the conceptual landscape of a given text using advanced language models like BERT, RoBERTa, and DistilBERT. The goal is to provide a user-friendly interface for analyzing and understanding the relationships between different concepts in your text.

There are a number of ways to run this repo.

## Installation of PyTorch
Due to the nature of PyTorch installations across different hardware, we have left the installation of PyTorch to the user. For installation instructions, please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/). The majority of the development for this repo was done using ROCm...not CUDA.

## User interface
You can run the code in this repo using the user interface. This is a simple web app that allows you to upload a text file and visualize the conceptual landscape. To run the user interface, follow these steps:

## Running larger experiments
`conceptual-cartography` also accepts config files that allow you to run larger experiments. These config files are located in the `configs` directory. Config files are in a YAML format and are configured in the following way:

```yaml
# This is currently a placeholder for the actual config file
# my_config.yaml
model: "bert-base-uncased"
text_dir: "data/texts"
output_dir: "data/output"
clustering:
  method: "kmeans"
  n_clusters: 5
  max_iter: 100
  random_state: 42
  n_init: 10
feature_extraction:
  method: "tf-idf"
  max_features: 1000
  ngram_range: [1, 2]
```
