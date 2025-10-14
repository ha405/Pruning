import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import json
from models.vgg_16_bn import get_model, get_model_100
from pruning.utils import get_execution_order, evaluate_accuracy
from profiling import profile_model


def run_sensitivity_analysis(model, testloader, device, dataset_name="cifar10", save_path="sensitivity_layerwise.csv", cifar100 = False):
    model = model.to(device)
    forward_order = get_execution_order(model)
    # Only Conv2d and Linear layers can be pruned
    to_prune = [layer for layer in forward_order if isinstance(layer, (nn.Conv2d, nn.Linear))]

    baseline_acc = evaluate_accuracy(model, testloader, device, max_batches=20)
    sparsity_levels = [0, 10, 20, 50, 70, 80, 90]
    results = []

    for layer in to_prune:
        layer_name = layer.__class__.__name__
        for sparse in sparsity_levels:
            if cifar100:
                pruned_model = get_model_100().to(device)
            else:
                pruned_model = get_model().to(device)
            pruned_model.load_state_dict(model.state_dict())

            target_layer = None
            for m in pruned_model.modules():
                if type(m) == type(layer) and hasattr(m, 'weight'):
                    target_layer = m
                    break

            if target_layer is None:
                continue

            W = target_layer.weight.data
            abs_w = torch.abs(W).view(-1)
            threshold = torch.quantile(abs_w, sparse / 100)
            mask = torch.abs(W) > threshold
            target_layer.weight.data = W * mask

            acc = evaluate_accuracy(pruned_model, testloader, device, max_batches=20)
            drop = baseline_acc - acc
            results.append({"layer": layer_name, "sparsity_pct": sparse, "top1": acc, "top1_drop": drop})

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    return df


def analyze_sensitivity(csv_path, dataset_name, model, target_sparsity=0.8, plot_drop=True):
    df = pd.read_csv(csv_path)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))

    for layer, group in df.groupby("layer"):
        x = group["sparsity_pct"]
        y = group["top1_drop"] if plot_drop else group["top1"]
        plt.plot(x, y, marker="o", label=layer)

    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy Drop (%)" if plot_drop else "Top-1 Accuracy (%)")
    plt.title(f"Layer-wise Sensitivity – {dataset_name}")
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"plots/sensitivity_{dataset_name}.png", dpi=200)
    plt.close()

    param_map = {}
    for name, p in model.named_parameters():
        if "weight" in name and not any(bn in name.lower() for bn in ["bn", "batchnorm"]):
            layer_name = name.rsplit('.', 1)[0]
            layer_type = type(dict(model.named_modules())[layer_name]).__name__
            param_map.setdefault(layer_type, []).append((name, p.numel()))

    results = []
    weighted_sum = 0
    total_params = sum([sum(x[1] for x in v) for v in param_map.values()])

    for layer, group in df.groupby("layer"):
        if any(bn in layer.lower() for bn in ["bn", "batchnorm"]):
            continue
        tolerable = group[group["top1_drop"] <= 12]
        best_sparse = tolerable["sparsity_pct"].max() if len(tolerable) > 0 else 20
        n_params = sum(x[1] for x in param_map.get(layer, []))
        weighted_sum += n_params * (best_sparse / 100.0)
        results.append({"layer": layer, "params": n_params, "assigned_sparsity": best_sparse})

    overall_sparsity = weighted_sum / total_params
    out_df = pd.DataFrame(results)
    os.makedirs("plans", exist_ok=True)
    out_df.to_csv(f"plans/sparsity_plan_{dataset_name}.csv", index=False)

    return overall_sparsity, out_df


def apply_pruning_masks_sparse(model, sparsity_plan_csv, dataset_name, target_sparsity, device="cpu"):
    model = model.to(device)
    sparsity_plan = pd.read_csv(sparsity_plan_csv)

    os.makedirs("results/pruning_masks", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/sparse_weights", exist_ok=True)

    masks = {}
    sparse_weights = {}

    for name, param in model.named_parameters():
        if "weight" not in name or any(bn in name.lower() for bn in ["bn", "batchnorm"]):
            continue

        layer_name = name.rsplit('.', 1)[0]
        layer_class = type(dict(model.named_modules())[layer_name]).__name__
        layer_row = sparsity_plan[sparsity_plan["layer"] == layer_class]
        if len(layer_row) == 0:
            continue

        sparsity_pct = float(layer_row["assigned_sparsity"].values[0])
        W = param.data
        abs_w = torch.abs(W).view(-1)
        threshold = torch.quantile(abs_w, sparsity_pct / 100)
        mask = (torch.abs(W) > threshold).to(W.device)
        param.data *= mask
        masks[name] = mask

        if W.ndim >= 2:
            if W.dim() == 2:
                sparse_weights[name] = W.to_sparse_csr()
            else:
                original_shape = W.shape
                W_flat_dense = W.reshape(original_shape[0], -1)
                W_sparse_flat_csr = W_flat_dense.to_sparse_csr()
                sparse_weights[name] = (W_sparse_flat_csr, original_shape)

    mask_path = f"results/pruning_masks/{dataset_name}_unstructured_mask.pt"
    model_path = f"results/models/{dataset_name}_vgg16_unstructured_{int(target_sparsity*100)}.pt"
    sparse_path = f"results/sparse_weights/{dataset_name}_sparse_weights.pt"

    torch.save(masks, mask_path)
    torch.save(model.state_dict(), model_path)
    torch.save(sparse_weights, sparse_path)

    print(f"Masks saved to {mask_path}")
    print(f"Pruned model saved to {model_path}")
    print(f"Sparse linear weights saved to {sparse_path}")

    return masks, sparse_weights


def verify_masks_coo(mask_path: str, sparse_path: str):
    masks = torch.load(mask_path)
    sparse_weights = torch.load(sparse_path)

    log = []
    for name in masks:
        mask = masks[name]
        sparse_object = sparse_weights.get(name) 
        if isinstance(sparse_object, tuple):
            W_sparse = sparse_object[0]
        else:
            W_sparse = sparse_object
        num_params = mask.numel()
        num_zeroed = (mask == 0).sum().item()
        sparse_nnz = W_sparse._nnz() if W_sparse is not None else 0
        matches = (num_params - num_zeroed) == sparse_nnz
        log.append({
            "layer": name,
            "num_params": num_params,
            "num_zeroed": num_zeroed,
            "sparse_nnz": sparse_nnz,
            "matches": matches
        })
    os.makedirs("results/verification", exist_ok=True)
    log_path = "results/verification/mask_csr_verification.csv"
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"Verification log saved to {log_path}")
    return log

def finetune_pruned_model(model, trainloader, val_loader, masks, device="cpu",
                          optimizer_type="sgd", lr=1e-3, momentum=0.9,
                          epochs=15, save_path=None):
    model = model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_state = None
    print("⚙️ Starting fine-tuning...")
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} started ===")
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_count += 1
            try:
                inputs, targets = inputs.to(device), targets.to(device)
            except Exception as e:
                print(f"[ERROR] Moving batch to device failed at batch {batch_idx}: {e}")
                continue

            optimizer.zero_grad()
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"[ERROR] Forward/backward/step failed at batch {batch_idx}: {e}")
                continue

            # Re-apply masks
            param_dict = dict(model.named_parameters())
            with torch.no_grad():
                for name, mask in masks.items():
                    if name in param_dict:
                        param_dict[name].data.mul_(mask)
                    else:
                        print(f"[WARNING] Mask for {name} not found in model params.")

            epoch_loss += loss.item()

            # Print every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(trainloader)} -> Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Batches: {batch_count}")

        # Validation
        model.eval()
        try:
            val_acc = evaluate_accuracy(model, val_loader, device)
        except Exception as e:
            print(f"[ERROR] Validation failed at epoch {epoch+1}: {e}")
            val_acc = 0.0

        print(f"Epoch {epoch+1}/{epochs} -> Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if save_path and best_state is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        print(f"Finetuned model saved to {save_path}")

    print(f"Best validation accuracy achieved: {best_acc:.2f}%")
    return best_acc


def profile_pruned_model(model, val_loader, device, dataset_name, save_path=None, num_batches=10):
    model = model.to(device)
    profile_results = profile_model(model, val_loader, dataset_name, device=device, num_batches=num_batches)
    os.makedirs("results", exist_ok=True)
    if save_path is None:
        save_path = f"results/pruned_profiling_{dataset_name}_unstructured.json"
    with open(save_path, "w") as f:
        json.dump(profile_results, f, indent=2)
    print(f"Profiling results saved to {save_path}")
    return profile_results
