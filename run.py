import json
import torch

from data.dataset import (
    cifar10_trainloader,
    ciaf10_testloader,
    cifar100_trainloader,
    ciaf100_testloader,
)
from models.vgg_16_bn import get_model
from profiling import profile_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    results = []

    print("\n CIFAR-10")
    model_cifar10 = get_modeL()
    testloader_10 = ciaf10_testloader(batch_size=64)
    result_cifar10 = profile_model(model_cifar10, testloader_10, "CIFAR-10", device)
    results.append(result_cifar10)

    print("\n CIFAR-100")
    model_cifar100 = get_modeL()
    testloader_100 = ciaf100_testloader(batch_size=64)
    result_cifar100 = profile_model(model_cifar100, testloader_100, "CIFAR-100", device)
    results.append(result_cifar100)

    with open("profiling_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n Profiling completed")


if __name__ == "__main__":
    main()
