from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

Transform = transforms.Compose([
    transforms.Resize((32, 32)),      
    transforms.ToTensor(),              
    transforms.Normalize(mean=mean, std=std)
])

def cifar10_trainloader(batch_size=64, shuffle=True):
    train = datasets.CIFAR10(root="./data", train=True, transform=Transform, download=True)
    return DataLoader(train, batch_size=batch_size, shuffle=shuffle)

def ciaf10_testloader(batch_size=64, shuffle=False):
    test = datasets.CIFAR10(root="./data", train=False, transform=Transform, download=True)
    return DataLoader(test, batch_size=batch_size, shuffle=shuffle)

def cifar100_trainloader(batch_size=64, shuffle=True):
    train = datasets.CIFAR100(root="./data", train=True, transform=Transform, download=True)
    return DataLoader(train, batch_size=batch_size, shuffle=shuffle)

def ciaf100_testloader(batch_size=64, shuffle=False):
    test = datasets.CIFAR100(root="./data", train=False, transform=Transform, download=True)
    return DataLoader(test, batch_size=batch_size, shuffle=shuffle)