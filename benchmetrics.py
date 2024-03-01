import argparse
import glob
import os

import pandas as pd
import seaborn as sns
import torch

from beexai.dataset.dataset import Dataset
from beexai.dataset.load_data import load_data
from beexai.evaluate.metrics.get_results import get_all_metrics
from beexai.explanation.explaining import CaptumExplainer
from beexai.training.train import Trainer
from beexai.utils.time_seed import set_seed
from beexai.utils.sampling import stratified_sampling

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="config/tabular_openml/clf_num",
    help="Path to folder containing config files",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="output/benchmarks",
    help="Path to folder to save results",
)
parser.add_argument(
    "--metrics",
    type=str,
    default="FaithCorr,Infidelity,Sensitivity,Comprehensiveness,Sufficiency,Monotonicity,AUC_TP,Complexity,Sparseness",
    help="Metrics to get",
)
parser.add_argument(
    "--methods",
    type=str,
    default="Lime,ShapleyValueSampling,KernelShap,DeepLift,IntegratedGradients,Saliency",
    help="Methods to use",
)

parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
parser.add_argument(
    "--n_sample", type=int, default=100, help="Number of samples to evaluate"
)

metric_obj = {
    "FaithCorr": "1-",
    "Infidelity": "0+",
    "Sensitivity": "0+",
    "Comprehensiveness": "1-",
    "Sufficiency": "0+",
    "Monotonicity": "1-",
    "AUC_TP": "0+",
    "Complexity": "0+",
    "Sparseness": "1-",
}
args = parser.parse_args()
SEED = args.seed
N_SAMPLE = args.n_sample
CONFIG_PATH = args.config_path
SAVE_PATH = args.save_path
METRICSTOGET = args.metrics.split(",")
METHODS = args.methods.split(",")
objectives = [metric_obj[x] for x in METRICSTOGET]

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

all_config_names = glob.glob(f"{CONFIG_PATH}/*.yaml")

metrics_objectives = [x + "_" + y for x, y in zip(METRICSTOGET, objectives)]
gradient_based = ["IntegratedGradients", "Saliency", "DeepLift"]

if not os.path.exists(f"{SAVE_PATH}/metrics"):
    os.makedirs(f"{SAVE_PATH}/metrics")
if not os.path.exists(f"{SAVE_PATH}/models"):
    os.makedirs(f"{SAVE_PATH}/models")
if not os.path.exists(f"{SAVE_PATH}/attributions"):
    os.makedirs(f"{SAVE_PATH}/attributions")

for path in all_config_names:
    DATA_NAME = path.split("/")[-1].split(".")[0].replace("\\", "_")
    data_test, target_col, task, _ = load_data(
        from_cleaned=True, config_path=path, keep_corr_features=True
    )
    if task == "classification":
        scale_params = {
            "x_num_scaler_name": "quantile_normal",
            "y_scaler_name": "labelencoder",
        }
    else:
        scale_params = {
            "x_num_scaler_name": "quantile_normal",
            "y_scaler_name": "minmax",
        }

    data = Dataset(data_test, target_col)
    X_train, X_test, y_train, y_test = data.get_train_test(
        test_size=0.2, scaler_params=scale_params
    )

    print(DATA_NAME, task, X_train.shape, y_train.shape)
    print(X_train.head())

    num_labels = data.get_classes_num(task)
    if task == "regression":
        BOOSTING_MODEL = "XGBRegressor"
    else:
        BOOSTING_MODEL = "XGBClassifier"

    cols = pd.MultiIndex.from_product(
        iterables=[metrics_objectives, ["NeuralNetwork", BOOSTING_MODEL]],
        names=["metrics", "model"],
    )
    if not os.path.exists(f"{SAVE_PATH}/metrics/{DATA_NAME}.csv"):
        dataset_df = pd.DataFrame(columns=cols)
        dataset_df["method"] = METHODS
        dataset_df.set_index("method", inplace=True)
        text = f"{DATA_NAME}_{task}_{X_train.shape[0]}_{X_train.shape[1]}_{num_labels}"
        dataset_df.index.name = text
    else:
        dataset_df = pd.read_csv(
            f"{SAVE_PATH}/metrics/{DATA_NAME}.csv", header=[0, 1], index_col=0
        )

    X_test, y_test = stratified_sampling(X_test, y_test, N_SAMPLE, task)
    print(y_test)
    for MODEL_NAME in [BOOSTING_MODEL, "NeuralNetwork"]:
        is_nn = MODEL_NAME == "NeuralNetwork"
        if MODEL_NAME == "NeuralNetwork":
            PARAMS = {
                "input_dim": X_train.shape[1],
                "output_dim": num_labels,
                "n_neurons": 128,
                "n_hidden_layers": 3,
                "batch_norm": True,
                "use_dropout": True,
            }
        else:
            PARAMS = {}
        trainer = Trainer(MODEL_NAME, task, PARAMS, device)

        if glob.glob(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}*"):
            if is_nn:
                trainer.load_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}.pt")
            else:
                trainer.load_model(
                    f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}.joblib"
                )
        else:
            trainer.train(
                X_train.values, y_train.values, learning_rate=0.001, epochs=1000
            )
            if MODEL_NAME == "NeuralNetwork":
                trainer.save_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}.pt")
                trainer.model.eval()
            else:
                trainer.save_model(
                    f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}.joblib"
                )

        perf_metric = trainer.get_metrics(X_test.values, y_test.values)
        print("Performance", perf_metric)
        methods = (
            [x for x in METHODS if x not in gradient_based] if not is_nn else METHODS
        )
        for METHOD in methods:
            print("Evaluate", METHOD, "for", MODEL_NAME, "on", DATA_NAME)
            USE_SKLEARN = MODEL_NAME != "NeuralNetwork"
            USE_ABS = task == "regression"
            XAI_DEVICE = device if not USE_SKLEARN else "cpu"
            exp = CaptumExplainer(trainer.model, task, METHOD, USE_SKLEARN, XAI_DEVICE)
            exp.init_explainer()

            metric = "accuracy" if task == "classification" else "mse"
            all_preds = (
                trainer.model.predict(X_test.values)
                if task == "classification"
                else None
            )
            attributions = exp.compute_attributions(
                X_test.values,
                data_name=DATA_NAME,
                model_name=MODEL_NAME,
                method_name=METHOD,
                folder_path=SAVE_PATH,
                preds=all_preds,
                save=True,
                use_abs=USE_ABS,
            )

            metric_df = get_all_metrics(
                X_test.values,
                all_preds,
                trainer.model,
                exp,
                auc_metric=metric,
                metrics_to_get=METRICSTOGET,
                attributions=attributions,
                device=XAI_DEVICE,
            )
            metric_df = metric_df.loc[~metric_df.index.duplicated(keep="first")]
            for i, metric in enumerate(METRICSTOGET):
                dataset_df.loc[METHOD, (metrics_objectives[i], MODEL_NAME)] = (
                    metric_df.loc[0, metric]
                )
        for metric in perf_metric:
            dataset_df.loc["Performance", (metric, MODEL_NAME)] = perf_metric[metric]
    cm = sns.light_palette("green", as_cmap=True)
    df_styled = dataset_df.style.background_gradient(
        cmap=cm,
        subset=[
            c for c in dataset_df.columns if c[1] in ["NeuralNetwork", BOOSTING_MODEL]
        ],
    )
    df_styled.to_html(f"{SAVE_PATH}/metrics/{DATA_NAME}.html")
    dataset_df.to_csv(f"{SAVE_PATH}/metrics/{DATA_NAME}.csv")
