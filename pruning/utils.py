import torch

def get_execution_order(model):
    execution_order = []

    def hook_fn(module, input, output):
        execution_order.append(module)

    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0: 
            hooks.append(module.register_forward_hook(hook_fn))
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, 224, 224).to(device)
    _ = model(dummy)

    for h in hooks:
        h.remove()

    return execution_order


def evaluate_accuracy(model, dataloader, device="cuda", max_batches=None):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if max_batches is not None and (i + 1) >= max_batches:
                break

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy
