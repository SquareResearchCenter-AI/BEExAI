# Usage

## Setup the configuration file

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
- target_col: target column of interest
- datetime_cols: columns with a datetime format that will be divided in several integer columns (year,month,day,hour)
- cols_to_delete: columns to drop (for example ID columns)
- cleaned_data_path: path to save the dataset after preprocessing, it can be directly used for repeated experiments
- task: classification or regression

Other operations such as adding specific colums from columns operations or deleting specific values must be done during the instanciation of the dataset in the notebooks or scripts.

## Notebooks

Several notebooks are available in the [notebook section](sequential) to train a model, compute explaination attributions and evaluation metrics on tabular data. It is recommended to execute the examples in the order they are presented.

## Load data and train a model

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

## Compute explaination attributions and evaluation metrics

```python
from beexai.explaining import CaptumExplainer
from beexai.metrics.get_results import get_all_metrics

METHOD = "IntegratedGradients"
exp = CaptumExplainer(trainer.model,task=task,method=METHOD,sklearn=False)
exp.init_explainer()

LABEL = 0
get_all_metrics(X_test.values,LABEL,trainer.model,exp)
```

