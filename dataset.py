from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def get_loader(batch_size=64, num_workers=0):
    # Training dataset
    train_loader = DataLoader(dataset=MNIST(root='.',
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,),
                                                                                               (0.3081,))])),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # Test dataset
    test_loader = DataLoader(dataset=MNIST(root='.',
                                           train=False,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,),
                                                                                              (0.3081,))])),
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader
