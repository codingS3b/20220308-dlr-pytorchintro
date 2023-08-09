import torch
import torchvision
from torchvision import datasets, transforms


def main(somepath="./pytorch-data"):

    # NOTE: the constants come from using the full training dataset for
    #       computation of the means and variances
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    normalized_dataset = datasets.MNIST(somepath, download=True, transform=transforms_)

    img, label = normalized_dataset[0]

    assert isinstance(img, torch.Tensor)
    assert img.mean() < 1.0
    assert img.mean() > 0.0
    print("first image in normalized MNIST:", img.mean(), img.std())
    # print("normalized batch mean value (fp32):", nbatch.float().mean())
    # print("normalized batch std  value (fp32):", torch.std(nbatch.float()))

    train_dataset = datasets.MNIST(
        somepath, download=True, transform=transforms_, train=True
    )
    test_dataset = datasets.MNIST(
        somepath, download=True, transform=transforms_, train=False
    )
    assert len(train_dataset) > len(test_dataset)
    assert len(train_dataset) == len(normalized_dataset)
    print(len(train_dataset), len(test_dataset))

    ## cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
        # **cuda_kwargs
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    print("our data sets for MNIST")
    print(train_dataset)
    print(test_dataset)

    for batch_idx, (X, y) in enumerate(train_loader):
        print("Batch", batch_idx, X.shape, y.shape)
        break


if __name__ == "__main__":
    main()
