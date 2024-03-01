from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from beexai.evaluate.metrics.auc_tp import compute_auc
from beexai.evaluate.metrics.complexity import compute_complex
from beexai.evaluate.metrics.comprehensiveness import compute_comp
from beexai.evaluate.metrics.faithfulnesscorr import compute_faith_corr
from beexai.evaluate.metrics.infidelity import compute_inf
from beexai.evaluate.metrics.monotonicity import compute_mono
from beexai.evaluate.metrics.sensitivity import compute_sens
from beexai.evaluate.metrics.sparseness import compute_spar
from beexai.evaluate.metrics.sufficiency import compute_suff
from beexai.explanation.explaining import GeneralExplainer
from beexai.utils.path import get_path

pd.set_option("display.max_columns", None)


def get_all_metrics(
    x_test: torch.Tensor,
    label: Optional[Union[int, list, np.ndarray, torch.Tensor]],
    model: Callable,
    exp: GeneralExplainer,
    ref_model: Optional[Callable] = None,
    refmodel_exp: Optional[GeneralExplainer] = None,
    baseline: str = "zero",
    auc_metric: str = "mse",
    subratio_faith: float = 0.2,
    comp_ratio: Union[float, list] = 0.3,
    suff_ratio: Union[float, list] = 0.3,
    inf_std: Optional[Union[float, torch.Tensor, np.ndarray]] = None,
    save_path: Optional[str] = None,
    metrics_to_get: List[str] = [
        "FaithCorr",
        "Infidelity",
        "Sensitivity",
        "Comprehensiveness",
        "Sufficiency",
        "Monotonicity",
        "AUC_TP",
        "Complexity",
        "Sparseness",
    ],
    print_plot: bool = False,
    attributions: Optional[torch.Tensor] = None,
    attributions_ref: Optional[torch.Tensor] = None,
    device: str = "cpu",
    use_ref: bool = False,
    use_random: bool = False,
    radius: Optional[float] = None,
) -> pd.DataFrame:
    """Compute all metrics for a given label.

    Args:
        x_test (torch.Tensor): test data
        label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
            Defaults to None. A list of labels can be provided, one for each instance.
        model (object): model to explain
        exp (object): explainer for the model to explain
        ref_model (object): reference model (random model)
        refmodel_exp (object): explainer for the reference model
        baseline (str, optional): baseline to use for the metrics.
            Defaults to "zero". Must be one of ["mean", "median", "zero",
            "multiple", "normal", "uniform"].
        auc_metric (str, optional): performance metric to use for the
            AUC_TP metric. Defaults to "mse". Must be one of
            ["mse","accuracy"].
        subratio_faith (float, optional): ratio of features to use for
            the faithfulness metric. Defaults to 0.2.
        comp_ratio (float, list, optional): ratio of features to remove
            for the comprehensiveness metric. Defaults to 0.3.
        suff_ratio (float, list, optional): ratio of features to keep for the
            sufficiency metric. Defaults to 0.3.
        inf_std (float, optional): std of the noise to add for
            the infidelity metric. Defaults to 0.003.
        save_path (str, optional): path to save the metrics.
            Defaults to None.
        metrics_to_get (list, optional): list of metrics to compute.
            Defaults to ["FaithCorr","Infidelity","Sensitivity",
            "Comprehensiveness","Sufficiency","Monotonicity","AUC_TP",
            "Complexity","Sparseness"].
        print_plot (bool, optional): whether to plot the figures and
            print the metrics. Defaults to False.
        attributions (torch.Tensor, optional): precomputed attributions
            for the model to explain. Defaults to None.
        attributions_ref (torch.Tensor, optional): precomputed
            attributions for the reference model. Defaults to None.
        device (str, optional): device to use. Defaults to "cpu".
        use_ref (bool, optional): whether to use the reference model
            for the metrics. Defaults to True.
        use_random (bool, optional): whether to use random attributions
            for the metrics. Defaults to True.
        radius (float, optional): radius for the sensitivity metric.
            Defaults to None.

    Returns:
        pd.DataFrame: dataframe containing the metrics
    """
    for metric in metrics_to_get:
        assert metric in [
            "FaithCorr",
            "Infidelity",
            "Sensitivity",
            "Comprehensiveness",
            "Sufficiency",
            "Monotonicity",
            "AUC_TP",
            "Complexity",
            "Sparseness",
        ], f"""Metric {metric} not recognized. Choose from:
            ["FaithCorr","Infidelity","Sensitivity","Comprehensiveness",
            "Sufficiency","Monotonicity","AUC_TP","Complexity","Sparseness"]"""
    if isinstance(x_test, pd.DataFrame):
        x_test = torch.tensor(x_test.values, dtype=torch.float32, device=device)
    if isinstance(x_test, np.ndarray):
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    if radius is None:
        radius = torch.mean(
            torch.stack(
                [
                    torch.abs(x_test[i] - x_test[j])
                    for i in range(x_test.shape[0])
                    for j in range(i + 1, x_test.shape[0])
                ]
            ),
            axis=0,
        )
    if inf_std is None:
        inf_std = torch.std(x_test, dim=0)
    metrics = defaultdict(dict)
    task = exp.task
    use_abs = task == "regression"
    subsize_faith = int(subratio_faith * x_test.shape[1])

    if attributions is not None:
        attributions = attributions.to(device)
    else:
        attributions = exp.explain(x_test, label=label, absolute=use_abs)
    orders = exp.feature_order(attributions)

    if use_ref:
        if attributions_ref is not None:
            randmodel_attributions = attributions_ref.to(device)
        else:
            randmodel_attributions = refmodel_exp.explain(
                x_test, label=label, absolute=use_abs
            )
        randmodel_orders = refmodel_exp.feature_order(randmodel_attributions)
    else:
        ref_model = None
        randmodel_attributions = None
        randmodel_orders = None
        refmodel_exp = None

    if use_random:
        lb, ub = torch.min(attributions), torch.max(attributions)
        rand_attrib = torch.rand(attributions.shape, device=device) * (ub - lb) + lb
        rand_orders = exp.feature_order(rand_attrib)
    else:
        rand_attrib = None
        rand_orders = None

    if "FaithCorr" in metrics_to_get:
        metrics = compute_faith_corr(
            model,
            ref_model,
            task,
            subsize_faith,
            x_test,
            attributions,
            rand_attrib,
            randmodel_attributions,
            label,
            metrics,
            device,
        )

    if "Infidelity" in metrics_to_get:
        metrics = compute_inf(
            model,
            ref_model,
            task,
            x_test,
            attributions,
            rand_attrib,
            randmodel_attributions,
            label,
            metrics,
            device,
            inf_std,
        )

    if "Sensitivity" in metrics_to_get:
        metrics = compute_sens(
            model,
            ref_model,
            task,
            x_test,
            label,
            metrics,
            exp,
            refmodel_exp,
            device,
            use_random,
            attributions,
            rand_attrib,
            randmodel_attributions,
            radius,
        )

    n_plot = x_test.shape[1] + 1
    if "Comprehensiveness" in metrics_to_get:
        metrics = compute_comp(
            model,
            ref_model,
            task,
            x_test,
            orders,
            rand_orders,
            randmodel_orders,
            n_plot,
            comp_ratio,
            label,
            metrics,
            baseline,
            print_plot,
            device,
        )

    if "Sufficiency" in metrics_to_get:
        metrics = compute_suff(
            model,
            ref_model,
            task,
            x_test,
            orders,
            rand_orders,
            randmodel_orders,
            n_plot,
            suff_ratio,
            label,
            metrics,
            baseline,
            print_plot,
            device,
        )

    if "Monotonicity" in metrics_to_get:
        metrics = compute_mono(
            model,
            ref_model,
            task,
            x_test,
            orders,
            rand_orders,
            randmodel_orders,
            label,
            metrics,
            baseline,
            device,
        )

    if "AUC_TP" in metrics_to_get:
        metrics = compute_auc(
            model,
            ref_model,
            task,
            x_test,
            orders,
            rand_orders,
            randmodel_orders,
            metrics,
            baseline,
            auc_metric,
            print_plot,
            device,
        )

    if "Complexity" in metrics_to_get:
        metrics = compute_complex(
            model,
            ref_model,
            task,
            attributions,
            rand_attrib,
            randmodel_attributions,
            metrics,
            device,
        )

    if "Sparseness" in metrics_to_get:
        metrics = compute_spar(
            model,
            ref_model,
            task,
            attributions,
            rand_attrib,
            randmodel_attributions,
            metrics,
            device,
        )

    comparators = ["Original"]
    if use_random:
        comparators.append("Random")
    if use_ref:
        comparators.append("Random Model")
    if len(comparators) > 1:
        cols = pd.MultiIndex.from_product([metrics_to_get, comparators])
    else:
        cols = metrics_to_get
    metrics = [list(metrics[metric].values()) for metric in metrics_to_get]
    metrics = np.array(metrics)
    metrics = np.reshape(metrics, (1, -1))
    df = pd.DataFrame(metrics, columns=cols)
    if print_plot:
        print("-------------------")
        print(df)
    if save_path is not None:
        save_path = get_path(save_path)
        df.to_csv(save_path)
    return df
