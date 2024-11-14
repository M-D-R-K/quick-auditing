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
    a user-defined test function before and after training.

    Parameters:
    - model: The model to be trained and audited.
    - device: The device (CPU/GPU) to run the model on.
    - test_loader: DataLoader for the test dataset.
    - criterion: Loss function.
    - train_func (Callable): User-defined training function that trains the model.
    - test_func (Callable): User-defined test function that evaluates the model and returns losses.

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
