import torch
## for the code to work, install torchvision
## $ python -m pip install --user -I --no-deps torchvision
import torchvision
from torchvision import datasets, transforms


def load_data(
    somepath,
    norm_loc=(0.1307,),  ## mu of normal dist to normalize by
    norm_scale=(0.3081,),  ## sigma of normal dist to normalize by
    train_kwargs={"batch_size": 64, "shuffle": True},
    test_kwargs={"batch_size": 1000},
    use_cuda=torch.cuda.device_count() > 0,
):
    """load MNIST data and return train/test loader object"""

    transform_ = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(norm_loc, norm_scale)]
    )

    train_dataset = datasets.MNIST(
        somepath, download=True, transform=transform_, train=True
    )
    test_dataset = datasets.MNIST(
        somepath, download=True, transform=transform_, train=False
    )

    if use_cuda:
        train_kwargs.update(
            {"num_workers": 1, "pin_memory": True, "shuffle": True})
        test_kwargs.update(
            {"num_workers": 1, "pin_memory": True, "shuffle": False})

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader


def main(somepath="./pytorch-data"):
    """load the data set and check the average and stddev of intensity (images are just tensors)"""

    train_loader, test_loader = load_data(somepath)

    for batch_idx, (X, Y) in enumerate(train_loader):
        print("train", batch_idx, X.shape, Y.shape)
        break

    for batch_idx, (X, Y) in enumerate(test_loader):
        print("test ", batch_idx, X.shape, Y.shape)
        break


if __name__ == "__main__":
    main()
    print("Ok. Checkpoint on loading data reached.")
