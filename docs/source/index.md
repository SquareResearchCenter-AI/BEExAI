# Welcome to BEExAI documentation!

[alt text]()

**BEExAI** is a Python library for benchmarking explainability methods on tabular data. It supports a wide range of explainability methods and evaluation metrics. It is designed to be easy to use and to allow fast obtention of benchmark results.

Major features include:
- Automatic preprocessing of tabular data
- Training of several models including [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/) Neural Network models.
- Computation of attributions for explainability methods from [Captum](https://captum.ai/)
- Computation of evaluation metrics for explainability methods for robustness, faithfulness and complexity

# Contents

```{toctree}
:maxdepth: 2
:caption: Introduction

installation
usage
technical_details
metrics
```

```{toctree}	
:caption: Examples
:maxdepth: 2

sequential
other
benchmark
```

```{toctree}
:caption: API Reference
:maxdepth: 2

api/add_api
api/modules
```

GitHub repository <https://github.com/SquareResearchCenter-AI/BEExAI>
