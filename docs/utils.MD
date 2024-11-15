# Utils Documentation
`utils.py` provides utility functions for privacy auditing and canary creation. It includes functions for finding the worst label for a data point, creating synthetic canaries, and computing estimated privacy parameters.

# Table of Contents

 - find_worst_label
 - create_canaries
 - get_e_hat
 - compute_e_hat
 
# Functions

## find_worst_label
`find_worst_label(` \
`data_point: torch.Tensor,` \
`model: Module,` \
`criterion: Module,` \
`labels: torch.Tensor) -> int`

Determines the label that maximizes the loss for a given data point.

**Parameters:**

-   `data_point` (torch.Tensor): The input data point.
-   `model` (Module): The model used to evaluate the data point.
-   `criterion` (Module): The loss function used to compute the loss.
-   `labels` (torch.Tensor): The possible labels to consider.

**Returns:**

-   `int`: The label with the highest loss

## create_canaries

`create_canaries( dataset: Dataset,` \
`num_canaries: int,` \
`canary_type: Optional[str] = None,`\
`model: Optional[Module] = None,`\
 `criterion: Optional[Module] = None`,\
  `device: Any = 'cpu'`\
  `)-> Tuple[TensorDataset, TensorDataset, torch.Tensor]`


Generates a dataset with canaries for privacy auditing by injecting synthetic samples.

**Parameters:**

-   `dataset` (Dataset): The original dataset.
-   `num_canaries` (int): The number of canaries to create.
-   `canary_type` (Optional[str]): The type of canary to create. Options include `"worst_label"`, `"random"`, and `"mislabeled"`.
-   `model` (Optional[Module]): A model for evaluating data points, used if `canary_type` is `"worst_label"`.
-   `criterion` (Optional[Module]): The loss criterion used for finding the worst label, required if `canary_type` is `"worst_label"`.
-   `device` (Any): The device (CPU/GPU) to run the model on.

**Returns:**

-   `Tuple[TensorDataset, TensorDataset, torch.Tensor]`:
    -   `train_dataset` (TensorDataset): The training dataset containing original data and included canaries.
    -   `canary_dataset` (TensorDataset): The dataset containing all canaries created.
    -   `S` (torch.Tensor): A tensor of 1s and -1s, indicating whether each canary was included (1) or not (-1).
## get_e_hat
`get_e_hat(N_correct: int, N_total: int) -> Union[float, int]`

Estimates the empirical epsilon for a differential privacy mechanism using the number of correct classifications out of a total, assuming a confidence level of 95%.

**Parameters:**

-   `N_correct` (int): The number of correctly classified samples.
-   `N_total` (int): The total number of samples.

**Returns:**

-   `Union[float, int]`: The estimated epsilon value. Returns `0` if the estimation fails.

## compute_e_hat

`compute_e_hat(`
    `losses_init: torch.Tensor,`\
    `losses_final: torch.Tensor,`\
   ` S: torch.Tensor,`\
    `get_e_hat: Callable[[int, int], float]`\
`) -> float`

Computes the maximum estimated epsilon (`e_hat`) based on the initial and final losses, inclusion vector `S`, and the provided `get_e_hat` function.

**Parameters:**

-   `losses_init` (torch.Tensor): Initial losses for each sample before training.
-   `losses_final` (torch.Tensor): Final losses for each sample after training.
-   `S` (torch.Tensor): Inclusion vector where 1 indicates a sample is in the canary set.
-   `get_e_hat` (Callable): Function to calculate epsilon based on `N_correct` and `N_total`.

**Returns:**

-   `float`: The maximum estimated epsilon value.