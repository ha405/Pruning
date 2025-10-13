import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.linear_model import Lasso

def collect_inputs_outputs(model, data, device):
    model.to(device)
    model.eval()
    conv_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    activations_in, activations_out = {}, {}
    def hook_fn(name):
        def hook(module, input, output):
            activations_in[name] = input[0].detach().cpu()
            activations_out[name] = output.detach().cpu()
        return hook
    hooks = [m.register_forward_hook(hook_fn(name)) for name, m in conv_layers]
    with torch.no_grad():
        for x, _ in data:
            _ = model(x.to(device)); break
    for h in hooks: h.remove()
    return activations_in, activations_out

def unfold_activations(activations, model):
    unf_x = {}
    named_modules = dict(model.named_modules())
    for name, x in activations.items():  
        layer = named_modules[name]
        x_unf = nn.functional.unfold(x, kernel_size=layer.kernel_size, padding=layer.padding, stride=layer.stride)
        unf_x[name] = x_unf.permute(0, 2, 1).reshape(-1, x_unf.shape[1])
    return unf_x

def get_channel_saliency_scores(model, unf_x, activations_out):
    saliency_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in activations_out:
            Y = activations_out[name].permute(0, 2, 3, 1).reshape(-1, module.out_channels).numpy()
            X_unf = unf_x[name].numpy()
            lasso = Lasso(alpha=1e-4, fit_intercept=False, precompute=True, positive=True, random_state=42, max_iter=1000)
            try: lasso.fit(X_unf, Y)
            except ValueError: continue
            betas = np.sum(np.abs(lasso.coef_), axis=0)
            patch_size = module.kernel_size[0] * module.kernel_size[1]
            channel_scores = np.array([np.sum(betas[i*patch_size:(i+1)*patch_size]) for i in range(module.in_channels)])
            saliency_scores[name] = channel_scores
    return saliency_scores

def reconstruct_and_prune_prev(model, unf_x, activations_out, layer_name, kept_indices, device):
    current_layer = model.get_submodule(layer_name).to(device)
    kh, kw = current_layer.kernel_size
    patch_size = kh * kw
    
    X_unf_t = torch.nan_to_num(unf_x[layer_name]).to(device)
    Y_t = torch.nan_to_num(activations_out[layer_name].permute(0, 2, 3, 1).reshape(-1, current_layer.out_channels)).to(device)
    
    cols = [i for ch_idx in kept_indices for i in range(ch_idx * patch_size, (ch_idx + 1) * patch_size)]
    X_prime = X_unf_t[:, cols]

    W_prime_T = torch.linalg.lstsq(X_prime, Y_t).solution
    W_prime = torch.nan_to_num(W_prime_T.T.reshape(current_layer.out_channels, len(kept_indices), kh, kw))
    
    current_layer.weight.data = W_prime
    current_layer.in_channels = len(kept_indices)

    all_modules = list(model.named_modules())
    current_layer_idx = [i for i, (name, _) in enumerate(all_modules) if name == layer_name][0]
    
    prev_conv_name, prev_bn_name = None, None
    for i in range(current_layer_idx - 1, -1, -1):
        name, module = all_modules[i]; 
        if isinstance(module, nn.Conv2d): prev_conv_name = name; break
    
    if prev_conv_name:
        for i in range(current_layer_idx - 1, -1, -1):
            name, module = all_modules[i]
            if name.startswith(prev_conv_name.rsplit('.',1)[0]) and isinstance(module, nn.BatchNorm2d):
                prev_bn_name = name; break
        
        prev_conv = model.get_submodule(prev_conv_name)
        prev_conv.weight.data = prev_conv.weight.data[kept_indices, :, :, :]
        if prev_conv.bias is not None: prev_conv.bias.data = prev_conv.bias.data[kept_indices]
        prev_conv.out_channels = len(kept_indices)
        
        if prev_bn_name:
            bn = model.get_submodule(prev_bn_name)
            bn.weight.data = bn.weight.data[kept_indices]
            bn.bias.data = bn.bias.data[kept_indices]
            bn.running_mean = bn.running_mean[kept_indices]
            bn.running_var = bn.running_var[kept_indices]
            bn.num_features = len(kept_indices)

def generate_plan_for_target_sparsity(csv_path, model, target_overall_sparsity=0.70):
    sensitivity_df = pd.read_csv(csv_path)
    prunable_modules = {name: m.weight.numel() for name, m in model.named_modules() if isinstance(m, (nn.Conv2d))}
    total_params = sum(prunable_modules.values())
    
    sensitivity_df['module_name'] = sensitivity_df['layer'].apply(lambda x: x.rsplit('.', 1)[0])
    
    base_plan = {name: 10.0 for name in prunable_modules.keys()}
    for name in base_plan:
        layer_df = sensitivity_df[sensitivity_df['module_name'] == name]
        if not layer_df.empty:
            tolerable = layer_df[layer_df['accuracy_drop'] <= 7.0]
            if not tolerable.empty: base_plan[name] = tolerable['sparsity_pct'].max()

    final_plan = deepcopy(base_plan)
    for _ in range(100):
        pruned_params = sum(prunable_modules.get(name, 0) * (final_plan.get(name, 0) / 100.0) for name in prunable_modules)
        sparsity = pruned_params / total_params
        if abs(sparsity - target_overall_sparsity) < 0.005: break
        gap = target_overall_sparsity - sparsity
        adjustment = gap * 2.0 
        for name in final_plan: final_plan[name] = min(max(final_plan[name] + adjustment, 0), 90.0)       
    return final_plan

@torch.no_grad()
def evaluate_accuracy(model, data_loader, device):
    model.eval(); correct, total = 0, 0
    for i, l in data_loader:
        o = model(i.to(device)); _, p = torch.max(o.data, 1)
        total += l.size(0); correct += (p == l.to(device)).sum().item()
    return 100 * correct / total

def calculate_structured_sparsity(o, p): return (sum(x.numel() for x in o.parameters()) - sum(x.numel() for x in p.parameters())) / sum(x.numel() for x in o.parameters())

def finetune_model(model, tr, te, device, epochs=40, lr=0.01):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for i, l in tr:
            opt.zero_grad(); crit(model(i.to(device)), l.to(device)).backward(); opt.step()
        sch.step()
        print(f"Epoch {epoch+1}/{epochs} — Accuracy: {evaluate_accuracy(model, te, device):.2f}%")
    return model