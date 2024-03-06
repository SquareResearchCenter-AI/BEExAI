# Add custom objects to the API

## Add custom explainer

To add a custom explainer, you need to create a class in `explaining.py` that inherits from the `GeneralExplainer` class and implements the following methods:
- `init_explainer`: initialize the explainer. All operations that need to be done before the computation of the attributions should be done here if your explainer needs it.
- `explain`: compute the attributions for the given data and model. The output should be a `torch.Tensor` of shape `(n_samples, n_features)` where `n_samples` is the number of samples in the data set and `n_features` is the number of features of said data set. In most cases, the input will be a `torch.Tensor` of shape `(n_samples, n_features)` with eventually the target class if needed.

```python
class MyExplainer(GeneralExplainer):
    def __init__(self, model, data):
        super().__init__(model, data)
        # Add eventually other parameters here

    def init_explainer(self, **kwargs):
        # Initialize the explainer here
        return explainer

    def explain(self, x_in, **kwargs):
        # Compute the attributions here
        return attributions
```

## Add custom metric

To add a custom metric, you need to create a module in `metrics` with a class that inherits from the `CustomMetric` class and implements at least a method `get_{metric_name}` where `metric_name` is the name of the metric. This method should take as input the computed attributions and other arguments if needed like the input data or the target class and return a `torch.Tensor` of shape `()` with the value of the metric, most likely averaged over the samples.

```python
class MyMetric(CustomMetric):
    def get_metric_name(self, attributions, **kwargs):
        # Compute the metric here
        return metric_value
```

## Examples

Here is an example of a custom explainer based on Captum library from which explainability methods are used in most of our benchmarks.

```python
class CaptumExplainer(GeneralExplainer):
    def __init__(
        self,
        model: callable,
        task: str,
        method: str,
        sklearn: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            model (callable): model to explain
            task (str): task to perform
            method (str): method to use
            sklearn (bool, optional): whether to use a sklearn model.
                Defaults to False.
            device (str, optional): device to use. Defaults to "cpu".
        """
        super().__init__(model, task)
        self.method = method
        self.sklearn = sklearn
        self.explainer = None
        self.all_methods = {
            "DeepLift": DeepLift,
            "IntegratedGradients": IntegratedGradients,
            "Saliency": Saliency,
            "ShapleyValueSampling": ShapleyValueSampling,
        }
        self.device = device

    def __forward_wrapper__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Wrapper for sklearn model to convert the input in tensors."""
        x_in = tensor.detach().cpu().numpy()
        if self.task == "regression":
            with torch.no_grad():
                output = self.model.predict(x_in)
        elif self.task == "classification":
            with torch.no_grad():
                output = self.model.predict_proba(x_in)
        return torch.from_numpy(output)

    def init_explainer(self) -> captum.attr.Attribution:
        """Initialize Captum explainer.

        Returns:
            captum.attr.Attribution: explainer
        """
        if self.sklearn:
            explainer = self.all_methods[self.method](self.__forward_wrapper__)
        else:
            explainer = self.all_methods[self.method](self.model)
        self.explainer = explainer
        return explainer

    def explain(self, x_test: pd.DataFrame, label: int = 0) -> torch.Tensor:
        """Explain the whole set.

        Args:
            x_test (pd.DataFrame): test set

        Returns:
            torch.Tensor: array of attributions (n_samples, n_features)
        """
        assert self.explainer is not None, "Explainer not initialized"
        x_tensor_test = x_test
        if isinstance(x_test, pd.DataFrame):
            x_tensor_test = torch.tensor(x_tensor_test.values)
        elif isinstance(x_test, np.ndarray):
            x_tensor_test = torch.tensor(x_tensor_test)
        if self.task == "regression":
            target = None
        elif self.task == "classification":
            target = label
        x_tensor_test = x_tensor_test.to(self.device)
        attributions = self.explainer.attribute(x_tensor_test, target=target)
        attributions = torch.abs(attributions).float()
        attributions = attributions / torch.norm(attributions)
        return attributions
```

Here is an example of a simple custom metric that computes the complexity of the attributions.

```python
class Complexity(CustomMetric):
    def __total_contribution__(self, attribution: torch.Tensor) -> torch.Tensor:
        """Compute the total contribution of each instance."""
        return torch.sum(torch.abs(attribution), axis=1)

    def __fractional_contribution__(
            self,
            attribution: torch.Tensor,
            feature_i: int
    ) -> torch.Tensor:
        """Compute the fractional contribution of a given feature"""
        total_contrib = self.__total_contribution__(attribution)
        return torch.abs(attribution[:, feature_i])/(total_contrib+1e-8)

    def get_cmpl(self, attribution: torch.Tensor) -> torch.Tensor:
        """Computes the complexity of the model.

        Args:
            attribution (torch.Tensor): attributions for each instance

        Returns:
            torch.Tensor: array of complexity scores for each instance
        """
        n_features = attribution.shape[1]
        complexity = torch.zeros(attribution.shape[0], device=self.device)
        for j in range(n_features):
            frac_contrib = self.__fractional_contribution__(attribution, j)
            complexity += - frac_contrib * torch.log(frac_contrib + 1e-8)
        complexity = complexity / n_features
        return torch.mean(complexity, axis=0)
```