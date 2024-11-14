import torch
from torch.utils.data import Dataset, TensorDataset
from torch.nn import Module
from typing import Optional, Tuple, List, Union, Callable

from scipy.stats import binom
from scipy.optimize import brentq

import math

def find_worst_label(
    data_point: torch.Tensor,
    model: Module,
    criterion: Module,
    labels: torch.Tensor
) -> int:
    """
    Determines the label that maximizes the loss for a given data point.

    Parameters:
    - data_point (torch.Tensor): The input data point.
    - model (Module): The model used to evaluate the data point.
    - criterion (Module): The loss function used to compute the loss.
    - labels (torch.Tensor): The possible labels to consider.

    Returns:
    - int: The label with the highest loss.
    """
    current_worst_loss = 0
    current_worst_label = -1
    for label in labels:
        output = model(data_point)
        loss = criterion(output, label).item()
        if loss > current_worst_loss:
            current_worst_loss = loss
            current_worst_label = label
    return current_worst_label



def create_canaries(
    dataset: Dataset,
    num_canaries: int,
    canary_type: Optional[str] = None,
    model: Optional[Module] = None,
    criterion: Optional[Module] = None,
    device: Any = 'cpu'
) -> Tuple[TensorDataset, TensorDataset, torch.Tensor]:
    """
    Generates a dataset with canaries for privacy auditing by injecting synthetic samples.

    Parameters:
    - dataset (Dataset): The original dataset.
    - num_canaries (int): The number of canaries to create.
    - canary_type (Optional[str]): The type of canary to create. Options include "worst_label", "random",
      and "mislabeled".
    - model (Optional[Module]): A model for evaluating data points, used if canary_type is "worst_label".
    - criterion (Optional[Module]): The loss criterion used for finding the worst label, required
      if canary_type is "worst_label".

    Returns:
    - Tuple[TensorDataset, TensorDataset, torch.Tensor]:
        - train_dataset (TensorDataset): The training dataset containing original data and included canaries.
        - canary_dataset (TensorDataset): The dataset containing all canaries created.
        - S (torch.Tensor): A tensor of 1s and -1s, indicating whether each canary was included (1) or not (-1).
    """

    # Initialize lists for fixed and canary data/labels
    fixed_data, fixed_labels = [], []
    canary_data, canary_labels = [], []
    
    if canary_type == "worst_label":
        # Create synthetic canaries with the label that maximizes the loss
        canary_data = torch.randn((num_canaries,) + dataset[0][0].shape)  # Generate random data points
        available_labels = torch.Tensor(dataset.targets).unique().long().unsqueeze(dim=-1)
        canary_labels = torch.Tensor([
            find_worst_label(canary_data[i], model, criterion, available_labels, device).item()
            for i in range(num_canaries)
        ]).long()

        # Collect remaining dataset samples as fixed data
        for idx, (data, label) in enumerate(dataset):
            if idx >= num_canaries:
                fixed_data.append(data)
                fixed_labels.append(label)

    elif canary_type == "random":
        # Generate random data and assign random labels from available labels
        canary_data = torch.randn((num_canaries,) + dataset[0][0].shape)
        possible_labels = torch.Tensor(dataset.targets).unique().long()
        canary_labels = torch.randint(0, len(possible_labels), (num_canaries,)).long()

    else:
        # Separate data from the dataset for canary or fixed usage
        for idx, (data, label) in enumerate(dataset):
            if idx < num_canaries:
                canary_data.append(data)
                canary_labels.append(label)
            else:
                fixed_data.append(data)
                fixed_labels.append(label)

        canary_data = torch.stack(canary_data).float()
        canary_labels = torch.Tensor(canary_labels).long()

        if canary_type == "mislabeled":
            # Randomly mislabel the canaries
            possible_labels = torch.unique(canary_labels)
            def mislabel(label):
                return possible_labels[possible_labels != label][torch.randint(0, len(possible_labels) - 1, (1,))].item()
            canary_labels = torch.Tensor([mislabel(label) for label in canary_labels]).long()

    # Stack the fixed data and labels if not using synthetic canaries
    if canary_type != "random":
        fixed_data = torch.stack(fixed_data).float()
        fixed_labels = torch.Tensor(fixed_labels).long()

    # Random inclusion of canaries
    S = torch.randint(low=0, high=2, size=(num_canaries,))
    S[S == 0] = -1  # Mark canaries for exclusion with -1

    # Include only the selected canaries
    included_data = canary_data[S == 1]
    included_labels = canary_labels[S == 1]

    # Concatenate fixed data with included canaries to create the training dataset
    train_data = torch.cat((fixed_data, included_data)) if canary_type != "random" else included_data
    train_labels = torch.cat((fixed_labels, included_labels)) if canary_type != "random" else included_labels

    train_dataset = TensorDataset(train_data, train_labels)
    canary_dataset = TensorDataset(canary_data, canary_labels)

    return train_dataset, canary_dataset, S



def get_e_hat(N_correct: int, N_total: int) -> Union[float, int]:
    """
    Estimates the empirical epsilon for a differential privacy mechanism using
    the number of correct classifications out of a total, assuming a confidence level of 95%.

    Parameters:
    - N_correct (int): The number of correctly classified samples.
    - N_total (int): The total number of samples.

    Returns:
    - Union[float, int]: The estimated epsilon value. Returns 0 if the estimation fails.
    """

    def binom_prob(epsilon: float) -> float:
        """
        Computes the difference between the cumulative probability of a binomial
        distribution with a probability parameter derived from epsilon and the target
        cumulative probability (0.95).

        Parameters:
        - epsilon (float): The privacy parameter.

        Returns:
        - float: The difference from the target cumulative probability of 0.95.
        """
        p = np.exp(epsilon) / (1 + math.exp(epsilon))  # Calculate the probability from epsilon
        return binom.cdf(N_correct, N_total, p) - 0.95  # Difference from the target probability

    try:
        # Use Brent's method to find the root of binom_prob in the interval [0, 50]
        epsilon_hat = brentq(binom_prob, 0, 50)
    except Exception as e:
        # Return 0 if root-finding fails
        epsilon_hat = 0

    return epsilon_hat



def compute_e_hat(
    losses_init: torch.Tensor,
    losses_final: torch.Tensor,
    S: torch.Tensor,
    get_e_hat: Callable[[int, int], float]
) -> float:
    """
    Computes the maximum estimated epsilon (`e_hat`) based on the initial and final losses,
    inclusion vector `S`, and the provided `get_e_hat` function.

    Parameters:
    - losses_init (torch.Tensor): Initial losses for each sample before training.
    - losses_final (torch.Tensor): Final losses for each sample after training.
    - S (torch.Tensor): Inclusion vector where 1 indicates a sample is in the canary set.
    - get_e_hat (Callable): Function to calculate epsilon based on N_correct and N_total.

    Returns:
    - float: The maximum estimated epsilon value.
    """

    # Calculate the score tensor as the difference between initial and final losses
    Scores = torch.stack((losses_init - losses_final, S), dim=1)
    m = len(Scores)  # Infer the number of canaries from Scores length

    e_hats = []

    # Determine step size based on `m`
    if m in [10, 50, 100]:
        step = 2
    elif m == 1000:
        step = 50
    else:
        step = max(1, m // 50)

    # Loop through possible m1 values to compute e_hats
    for m1 in range(1, m // 2, step):
        m2 = m1 if m not in [10, 50, 100] else m1

        # Sort the scores and apply the appropriate values for m1 and m2
        sorted_tensor, indices = torch.sort(Scores[:, 0], descending=False)
        sorted_tensor = Scores[indices]
        sorted_tensor[:m1, 0] = -1
        sorted_tensor[m1:-m2, 0] = 0
        sorted_tensor[-m2:, 0] = 1

        # Calculate the multiplication result and filter to correct instances
        multiplication_result = sorted_tensor[:, 0] * sorted_tensor[:, 1]
        result = torch.maximum(multiplication_result, torch.tensor(0))

        # Calculate N_correct and N_total for e_hat
        N_correct = sum(result).item()
        N_total = m1 + m2

        # Append the computed epsilon for this iteration
        e_hats.append(get_e_hat(N_correct, N_total))

    # Return the maximum epsilon estimate
    return max(e_hats)
