import os
import time
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torchprofile import profile_macs

try:
    from pyJoules.energy_meter import measure_energy
    from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
    from pyJoules.device.nvidia_device import NvidiaGPUDomain
    PYJOULES_AVAILABLE = True
except ImportError:
    PYJOULES_AVAILABLE = False
    print("pyJoules not installed — energy profiling will be skipped.")


def get_model_size_mb(model, temp_path="temp_model.pt"):
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 ** 2)
    os.remove(temp_path)
    return round(size_mb, 3)


def get_macs(model, batch_size=1, img_size=224, device="cuda"):
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    macs = profile_macs(model, dummy_input)
    return round(macs / 1e6, 3)  # in Millions


def measure_inference_latency(model, dataloader, device, num_batches=10):
    model.eval()
    latencies, peak_mem, avg_mem = [], [], []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            start = time.perf_counter()

            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                         record_shapes=False, profile_memory=True) as prof:
                outputs = model(inputs)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latency = (end - start) * 1000  # ms
            latencies.append(latency)
            peak_mem.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            avg_mem.append(torch.cuda.memory_allocated(device) / (1024 ** 2))

    avg_latency = round(sum(latencies) / len(latencies), 3)
    avg_peak_mem = round(sum(peak_mem) / len(peak_mem), 3)
    avg_used_mem = round(sum(avg_mem) / len(avg_mem), 3)

    return avg_latency, avg_peak_mem, avg_used_mem


def evaluate_accuracy(model, dataloader, device, topk=(1, 5)):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()

            _, pred_top5 = outputs.topk(5, 1, True, True)
            correct_top5 += sum([labels[i] in pred_top5[i] for i in range(len(labels))])
            total += labels.size(0)

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    return round(top1, 3), round(top5, 3)


def measure_energy_model(model, dataloader, device, num_batches=10):
    if not PYJOULES_AVAILABLE:
        return None
    energy_records = []
    try:
        @measure_energy(domains=[RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)])
        def inference_batch(inputs):
            with torch.no_grad():
                _ = model(inputs)
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                inputs = inputs.to(device)
                inference_batch(inputs)
                energy_records.append(getattr(inference_batch, "energy_consumed", None))
        energy_vals = [e for e in energy_records if e is not None]
        if len(energy_vals) == 0:
            return None
        avg_energy = sum(energy_vals) / len(energy_vals)
        return round(avg_energy, 3)
    except Exception:
        return None



def profile_model(model, dataloader, dataset_name, device="cuda", num_batches=10):
    print(f"\n Profiling {model.__class__.__name__} on {dataset_name}...")

    model = model.to(device)
    model.eval()

    # Step 1 — Model Size
    size_mb = get_model_size_mb(model)

    # Step 2 — MACs
    macs_million = get_macs(model, batch_size=1, device=device)

    # Step 3 — Latency + Memory
    latency_ms, peak_mem_mb, avg_mem_mb = measure_inference_latency(
        model, dataloader, device, num_batches
    )

    # Step 4 — Energy
    energy_mj = measure_energy_model(model, dataloader, device, num_batches)

    # Step 5 — Accuracy
    top1, top5 = evaluate_accuracy(model, dataloader, device)

    # Combine results
    results = {
        "Model": model.__class__.__name__,
        "Dataset": dataset_name,
        "Size (MB)": size_mb,
        "MACs (M)": macs_million,
        "Peak Mem (MB)": peak_mem_mb,
        "Avg Mem (MB)": avg_mem_mb,
        "Latency (ms)": latency_ms,
        "Energy (mJ)": energy_mj,
        "Top-1 (%)": top1,
        "Top-5 (%)": top5,
    }

    print("\n Profiling Results:")
    for k, v in results.items():
        print(f"{k:<15}: {v}")

    return results
