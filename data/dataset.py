from torchvision import datasets, transforms
from torch.utils.data import DataLoader

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std  = [0.2470, 0.2435, 0.2616]

cifar100_mean = [0.5071, 0.4865, 0.4409]
cifar100_std  = [0.2673, 0.2564, 0.2762]

cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
])
cifar10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
])

cifar100_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
])
cifar100_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
])

def cifar10_trainloader(batch_size=64, shuffle=True):
    train = datasets.CIFAR10(root="./data", train=True, transform=cifar10_train_transform, download=True)
    return DataLoader(train, batch_size=batch_size, shuffle=shuffle)

def ciaf10_testloader(batch_size=64, shuffle=False):
    test = datasets.CIFAR10(root="./data", train=False, transform=cifar10_test_transform, download=True)
    return DataLoader(test, batch_size=batch_size, shuffle=shuffle)

def cifar100_trainloader(batch_size=64, shuffle=True):
    train = datasets.CIFAR100(root="./data", train=True, transform=cifar100_train_transform, download=True)
    return DataLoader(train, batch_size=batch_size, shuffle=shuffle)

def ciaf100_testloader(batch_size=64, shuffle=False):
    test = datasets.CIFAR100(root="./data", train=False, transform=cifar100_test_transform, download=True)
    return DataLoader(test, batch_size=batch_size, shuffle=shuffle)
