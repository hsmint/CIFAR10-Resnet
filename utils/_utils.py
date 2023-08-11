from torchvision import datasets, transforms
from torch.utils.data import DataLoader

custom_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def make_data_loader(args):
    train_dataset = datasets.CIFAR10(args.data, train=True, transform=custom_transform, download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=custom_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader
