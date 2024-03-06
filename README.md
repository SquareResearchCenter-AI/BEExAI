<a name="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/SquareResearchCenter-AI/BEExAI">
  </a>

<h3 align="center">BEExAI</h3>

  <p align="center">
    Benchmark to Evaluate EXplainable AI
    <br />
    <a href="https://beexai.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SquareResearchCenter-AI/BEExAI/issues">Report Bug</a>
    ·
    <a href="https://github.com/SquareResearchCenter-AI/BEExAI/issues">Request Feature</a>
  </p>
</div>

[![PyPI](https://img.shields.io/pypi/v/beexai)](https://pypi.org/project/beexai/)
![PyPI](https://img.shields.io/pypi/v/beexai?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/beexai)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/beexai)](https://pypi.org/pypi/beexai/)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#technical-aspects">Technical aspects</a></li>
        <ul>
        <li><a href="#content-description">Content description</a></li>
        <li><a href="#supported-models">Supported models</a></li>
        <li><a href="#supported-explainability-methods">Supported explainability methods</a></li>
        <li><a href="#implemented-metrics">Implemented metrics</a></li>
        <li><a href="#disclaimer">Disclaimer</a></li>
        </ul>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project provides simple tools to benchmark multiple explainable AI methods on multiples Machine Learning models with customizable datasets and compute metrics to evaluate these methods. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

The project is entirely made in Python 3.9 and tested on Windows 11 64 bits. Both CPU and GPU are supported with PyTorch 2.0.1.

### Installation

BEExAI can be installed from [PyPI](https://pypi.org/) with:
```
pip install beexai
```

You can also install the project from source using:

1. Clone the repo
   ```sh
   git clone https://github.com/SquareResearchCenter-AI/BEExAI.git
   ```
2. Install the requirements
   ```sh
   cd BEExAI
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Setup a config 

To train a model, compute explaination attributions and evaluation metrics on tabular data, you will need to specify a config file for each dataset. There are several examples in `config/` with the following format:
```yaml
path: "data/my_dataset.csv"
target_col: "class"
datetime_cols: 
    - "date"
cols_to_delete:
    - "ID"
cleaned_data_path: "output/data/my_dataset_cleaned.csv"
task: "classification"
```
The different options can be described as follow:
- path: path of the dataset, it can be usually placed in a folder `data/`
- target_col: target column for training
- datetime_cols: columns with a datetime format that will be divided in several integer columns (year,month,day,hour)
- cols_to_delete: columns to drop (for example ID columns)
- cleaned_data_path: path to save the dataset after preprocessing for repeated usage, usually in `output/data`
- task: classification or regression
  
Other operations such as adding specific colums from columns operations or deleting specific values must be done during the instanciation of the dataset in the notebooks or scripts.

### Notebooks

Several notebooks are available in `notebooks/` for simple use cases:
- The numeroted serie can be ran in the order with your own dataset or with the examples provided (kickstarter and boston-credit dataset). 
- `all_in_one.ipynb` synthesizes the 3 notebooks in a single one without the detailed explanations.

### Load data and train model

```python
from beexai.dataset.load_data import load_data
from beexai.dataset.dataset import Dataset
from beexai.training.train import Trainer

DATA_NAME = "configname"
MODEL_NAME = "NeuralNetwork"
CONFIG_PATH = f"config/{DATA_NAME}.yml"

df,target_col,task,_ = load_data(from_cleaned=False,config_path=CONFIG_PATH)
data = Dataset(df,target_col)
X_train, X_test, y_train, y_test = data.get_train_test()

NN_PARAMS = {"input_dim":X_train.shape[1],"output_dim":num_labels}
trainer = Trainer(MODEL_NAME,task,NN_PARAMS)
trainer.train(X_train, y_train)
```

### Compute explanability metrics

```python
from beexai.explaining import CaptumExplainer
from beexai.metrics.get_results import get_all_metrics

METHOD = "IntegratedGradients"
exp = CaptumExplainer(trainer.model,task=task,method=METHOD,sklearn=False)
exp.init_explainer()

get_all_metrics(X_test.values,LABEL,trainer.model,exp)
```

_For more examples, please refer to the [Documentation](https://beexai.readthedocs.io/en/latest/)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Download datasets

The datasets used in this benchmarks are issued from several openml suites. 

The ones from [Why do tree-based models still outperform deep
learning on typical tabular data?](https://huggingface.co/datasets/inria-soda/tabular-benchmark) are the suites with ID 297,298,299 and 304.

The ones for multiclass classification are from tasks 12,14,16,18,22,23,28 and 32.

A simplified script to download them with OpenML API and create their configuration files is available in the root folder.

```sh
python openml_download.py
```

### Run benchmarks 

Running benchmarks can be done with the script `benchmetrics.py` with multiple arguments:
```sh
python benchmetrics.py --config_path config_folder --save_path output/my_benchmark --seed 42 --n_sample 1000
```

For comparison with the benchmarks in the `benchmark_results` folder, we used 1000 samples from the test set.

<!-- TECHNICAL ASPECts -->
## Technical aspects

### Content description

- benchmark_results: Complete benchmark results from our paper `insert_link` averaged on 5 random seeds
- config: Please detail here some basic information on your data. Other more complex operations on your data need to be done directly in the notebooks or scripts
- data: [boston](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset) and [kickstarter](https://www.kaggle.com/datasets/kemical/kickstarter-projects) datasets from [Kaggle](https://www.kaggle.com)
- notebooks: Simple use cases in notebook format
- output: Store outputs such as cleaned datasets, saved models and computed attributions
- src: Python scripts with main classes

### Supported models

- Linear Regression, Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- XGBoost
- Dense Neural Network

### Supported explainability methods

- Perturbation based: FeatureAblation, Lime, ShapleyValueSampling, KernelShap
- Gradient based: Integrated Gradients, Saliency, DeepLift, InputXGradient

### Implemented metrics

- Robustness: Sensitivity
- Faithfulness: Infidelity, Comprehensiveness, Sufficiency, Faithfulness Correlation, AUC-TP, Monotonicity
- Complexity: Complexity, Sparseness

### Disclaimer
The proposed pipeline might not include all possible customizations (especially for data preprocessing), feel free to add your own processing within the example notebooks.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

BEExAI is open-sourced with BSD-3 license.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Project Link: [https://github.com/SquareResearchCenter-AI/BEExAI](https://github.com/SquareResearchCenter-AI/BEExAI)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Square Management](https://www.square-management.com/)
* [Square Research Center](https://www.square-management.com/square-research-center/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SquareResearchCenter-AI/BEExAI.svg?style=for-the-badge
[contributors-url]: https://github.com/SquareResearchCenter-AI/BEExAI/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SquareResearchCenter-AI/BEExAI.svg?style=for-the-badge
[forks-url]: https://github.com/SquareResearchCenter-AI/BEExAI/network/members
[stars-shield]: https://img.shields.io/github/stars/SquareResearchCenter-AI/BEExAI.svg?style=for-the-badge
[stars-url]: https://github.com/SquareResearchCenter-AI/BEExAI/stargazers
[issues-shield]: https://img.shields.io/github/issues/SquareResearchCenter-AI/BEExAI.svg?style=for-the-badge
[issues-url]: https://github.com/SquareResearchCenter-AI/BEExAI/issues
[license-shield]: https://img.shields.io/github/license/SquareResearchCenter-AI/BEExAI.svg?style=for-the-badge
[license-url]: https://github.com/SquareResearchCenter-AI/BEExAI/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
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
