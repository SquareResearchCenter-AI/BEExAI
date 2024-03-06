# Benchmark

Benchmarks can be run in command line with several options to indicate dataset information and other parameters.
```
python benchmetrics.py --config_path config_folder --save_path output/my_benchmark --seed 42 --n_sample 1000
```
`config_folder` needs to have all configuration files at the top-level in YAML format.

## Download datasets from OpenML

To reproduce the benchmark results, you will need to download the datasets from OpenML.

The datasets used in this benchmarks are issued from several [openml](https://www.openml.org/) suites. 

The ones from [`Why do tree-based models still outperform deep learning on typical tabular data?`](https://huggingface.co/datasets/inria-soda/tabular-benchmark) are the suites with ID 297,298,299 and 304.

The ones for multiclass classification [OpenML-CC18 Curated Classification Benchmark](https://www.openml.org/search?type=benchmark&study_type=task&id=99) are from tasks 12,14,16,18,22,23,28 and 32.

A simplified script to download them with OpenML API and create their configuration files is available in the root folder.

```bash
python openml_download.py
```

## Benchmark results

All results of our benchmarks can be found in the folder `benchmark_results` available at this [link](https://github.com/SquareResearchCenter-AI/BEExAI/tree/main/benchmark_results). We invite to read our research paper for more details about these results and our analysis.
