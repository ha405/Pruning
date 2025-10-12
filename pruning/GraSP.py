import torch
import torch.nn as nn

def saliency_scores(model, data, device, loss_fn):
    model.eval()
    scores = {}
    count = 0

    for (x, y) in data:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True, retain_graph=True)
        params = [p for p in model.parameters() if p.requires_grad]
        try:
            Hv = torch.autograd.grad(
                grads,
                params,
                grad_outputs=[g.detach() if (g is not None) else torch.zeros_like(p) for g, p in zip(grads, params)],
                retain_graph=False,
                create_graph=False,
            )
        except RuntimeError:
            Hv = tuple(torch.zeros_like(p) for p in params)

        for (name, param), hv in zip(((n,p) for n,p in model.named_parameters() if p.requires_grad), Hv):

            hv = hv if hv is not None else torch.zeros_like(param)
            this_score = -param.detach() * hv.detach()
            if name not in scores:
                scores[name] = this_score.clone()
            else:
                scores[name] += this_score
        count += 1

    if count == 0:
        raise RuntimeError("No data seen in saliency_scores.")
    for k in list(scores.keys()):
        s = scores[k] / float(count)
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, float('-1e30')))
        scores[k] = s

    return scores


def rank_by_saliency(scores, current_mask=None, current_sparsity=0.0, target_sparsity=0.8):

    total_params = sum([s.numel() for s in scores.values()])
    num_prune_total = int(round(target_sparsity * total_params))
    num_already_pruned = int(round(current_sparsity * total_params))
    num_to_prune_now = num_prune_total - num_already_pruned
    if num_to_prune_now <= 0:
        print(f"[rank_by_saliency] Target sparsity {target_sparsity*100:.1f}% already reached or nothing to prune.")
        return (current_mask if current_mask is not None else {k: torch.ones_like(v) for k,v in scores.items()}), None
    live_scores_list = []
    names_and_masks = []
    for name, s in scores.items():
        base_mask = current_mask[name].bool() if (current_mask is not None and name in current_mask) else torch.ones_like(s).bool()
        live = base_mask
        names_and_masks.append((name, s, base_mask))
        if live.sum().item() > 0:
            live_scores_list.append(torch.abs(s[live]).view(-1))

    if len(live_scores_list) == 0:
        raise RuntimeError("[rank_by_saliency] No live parameters to prune.")

    live_scores = torch.cat(live_scores_list)
    num_live = live_scores.numel()

    if num_to_prune_now >= num_live:
        if target_sparsity < 0.999:
            print(f"[rank_by_saliency] Requested to prune {num_to_prune_now} but only {num_live} live. Capping to keep 1 live param.")
            num_to_prune_now = max(0, num_live - 1)
        else:
            num_to_prune_now = num_live  

    if num_to_prune_now == 0:
        print("[rank_by_saliency] After capping, nothing to prune this step.")
        return (current_mask if current_mask is not None else {k: torch.ones_like(v) for k,v in scores.items()}), None

    kth = num_to_prune_now
    vals, _ = torch.kthvalue(live_scores, kth)
    threshold = vals.item()  

    mask_dict = {}
    for name, s, base_mask in names_and_masks:
        base_mask = base_mask.to(s.device).float()
        live_idx = base_mask.bool()
        new_mask = base_mask.clone()
        if live_idx.sum().item() > 0:
            keep_live = (torch.abs(s[live_idx]) > threshold)
            new_mask[live_idx] = keep_live.float()
        mask_dict[name] = new_mask.to(s.device).type_as(s)
    total_kept = sum([m.sum().item() for m in mask_dict.values()])
    total_masked = total_params - total_kept
    print(f"[rank_by_saliency] Pruned {int(total_masked)} / {int(total_params)} (target {int(num_prune_total)}). Kept {int(total_kept)} params. threshold={threshold:.6g}")

    return mask_dict, threshold


def apply_mask(model, mask_dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                mask = mask_dict[name].to(param.device).type_as(param)
                if mask.shape != param.shape:
                    raise RuntimeError(f"Mask shape mismatch for {name}: mask {mask.shape} vs param {param.shape}")
                param.mul_(mask)
