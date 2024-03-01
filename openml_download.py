import os

import numpy as np
import openml
import yaml

save_path = "output/data/tabular_openml/"
config_path = "config/tabular_openml/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(config_path):
    os.makedirs(config_path)

# Download all datasets for https://arxiv.org/abs/2207.08815
task_ids = {"clf_num": 298, "reg_num": 297, "clf_cat": 304, "reg_cat": 299}
for task in task_ids:
    benchmark_suite = openml.study.get_suite(suite_id=task_ids[task])
    print(benchmark_suite.data)

    if not os.path.exists(f"{save_path}{task}"):
        os.makedirs(f"{save_path}{task}")
    if not os.path.exists(f"{config_path}{task}"):
        os.makedirs(f"{config_path}{task}")

    for id in benchmark_suite.data:
        dataset = openml.datasets.get_dataset(
            id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        name = dataset.name
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        data = X.copy()
        data[dataset.default_target_attribute] = y
        data.to_csv(f"{save_path}{task}/{name}.csv", index=False)

        print(f"Dataset {id} has {X.shape[0]} samples and {X.shape[1]} features")

        config = {
            "path": f"{save_path}{task}/{name}.csv",
            "target_col": dataset.default_target_attribute,
            "cleaned_data_path": f"{save_path}{task}/{name}.csv",
            "task": "regression" if "reg" in task else "classification",
        }
        with open(f"{config_path}{task}/{name}.yaml", "w") as file:
            yaml.dump(config, file)

# Download all datasets for OpenML-CC18 with ID 12,14,16,18,22,23,28 and 32
task_ids = [12, 14, 16, 18, 22, 23, 28, 32]
if not os.path.exists(f"{save_path}multi_class"):
    os.makedirs(f"{save_path}multi_class")
if not os.path.exists(f"{config_path}multi_class"):
    os.makedirs(f"{config_path}multi_class")

for id in task_ids:
    task = openml.tasks.get_task(id)
    dataset = openml.datasets.get_dataset(
        task.dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    name = dataset.name
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=task.target_name)
    data = X.copy()
    data[task.target_name] = y
    data.to_csv(f"{save_path}multi_class/{name}.csv", index=False)

    print(
        f"Dataset {id} has {X.shape[0]} samples, {X.shape[1]} features \
          and {len(np.unique(y))} classes"
    )

    config = {
        "path": f"{save_path}multi_class/{name}.csv",
        "target_col": task.target_name,
        "cleaned_data_path": f"{save_path}multi_class/{name}.csv",
        "task": "classification",
    }
    with open(f"{config_path}multi_class/{name}.yaml", "w") as file:
        yaml.dump(config, file)
