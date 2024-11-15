# audit Documentation
he `audit.py` file provides functionality to audit the training procedure of a model by recording initial and final losses on the canary dataset based upon the black box auditing method used by Thomas et al. in "Privacy Auditing in One Training Run". 

# Table of Contents

 - audit_training
 
# Functions

## audit_training
`audit_training(`\
`model: Any,`\
`device: Any,`\
`train_loader: Any,`\
`optimizer : Any,`\
`test_loader: Any,`\
`canary_loader : Any,`\
`criterion: Any,`\
`train_func: Callable,`\
`test_func: Callable[[Any, Any, Any, Any], List[float]],`\
`epochs : int  =  5,`\
`) -> Tuple[List[float], List[float]]`

Audits the training procedure by recording initial and final losses using a user-defined testing function before and after training. This function is designed to help measure changes in model performance or behavior resulting from training.

*
#### Parameters:

-   `model` (`Any`): The model to be trained and audited.
-   `device` (`Any`): The device (CPU/GPU) to run the model on.
-   `train_loader` (`Any`): DataLoader for the training dataset.
-   `optimizer` (`Any`): Optimizer used to update model parameters.
-   `test_loader` (`Any`): DataLoader for the test dataset.
-   `canary_loader` (`Any`): DataLoader specifically for canary samples, used to assess privacy impact.
-   `criterion` (`Any`): Loss function used to compute the loss.
-   `train_func` (`Callable`): User-defined function that trains the model. It should take arguments `(model, train_loader, optimizer, criterion, device, test_loader, epochs)`.
-   `test_func` (`Callable[[Any, Any, Any, Any], List[float]]`): User-defined function that evaluates the model and returns a list of loss values. It should take arguments `(model, loader, criterion, device)`.
-   `epochs` (`int`, optional): Number of training epochs. Defaults to 5.

#### Returns:

-   `Tuple[List[float], List[float]]`:
    -   `initial_losses`: A list of loss values recorded before training.
    -   `final_losses`: A list of loss values recorded after training.

### Points to Note:

 - The test function must accept parameters of the form `model, data_loader, criterion, device` and must output the losses for each sample separately. 
 For example,
 `def test(model, test_loader, criterion, device):`\
  `  model.eval()`\
 `   test_loss = 0.0`\
  `  correct = 0`\
 `   total = 0`\
`    losses = []`\
`    `
 `   with torch.no_grad():`\
`        for data, target in test_loader:`\
 `           data, target = data.to(device), target.to(device)`\
`            output = model(data)`\
 `           test_loss += criterion(output, target).item()  # Sum up batch loss`\
`            losses.append(criterion(output, target).item())`\
 `           _, predicted = output.max(1)`\
`            correct += predicted.eq(target).sum().item()`\
 `           total += target.size(0)`\
``\
 `   # Calculate average test loss and accuracy`\
`    test_loss /= len(test_loader)`\
 `   test_accuracy = 100. * correct / total`\
``
`   print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")`\
`    return test_loss, test_accuracy, losses`