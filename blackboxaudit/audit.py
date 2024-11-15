from typing import List, Callable, Tuple, Any

def audit_training(
    model: Any,
    device: Any,
    train_loader: Any,
    optimizer : Any,
    test_loader: Any,
    canary_loader : Any,
    criterion: Any,
    train_func: Callable,
    test_func: Callable[[Any, Any, Any, Any], List[float]],
    epochs : int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Audits the training procedure by recording initial and final losses using
    a the user defined test and train functions before and after training. 
    Note: the test and traing function must input and return outputs in the same format as given below.

    Parameters:
    - model: The model to be trained and audited.
    - device: The device (CPU/GPU) to run the model on.
    - train_loader: DataLoader for the train dataset
    - test_loader: DataLoader for the test dataset.
    - canary_loader: DataLoader for the canary dataset.
    - criterion: Loss function.
    - train_func (Callable): User-defined training function that trains the model.
    - test_func (Callable): User-defined test function that evaluates the model and returns per canary sample losses.

    Returns:
    - Tuple[List[float], List[float]]: Initial losses before training and final losses after training.
    """

    # Record initial losses before training
    initial_losses = test_func(model, canary_loader, criterion, device)
    print("Initial losses recorded.")

    # Run the user-defined training function
    train_func(model, train_loader, optimizer, criterion, device, test_loader, epochs)
    print("Training complete.")

    # Record final losses after training
    final_losses = test_func(model, canary_loader, criterion, device)
    print("Final losses recorded.")

    return initial_losses, final_losses
