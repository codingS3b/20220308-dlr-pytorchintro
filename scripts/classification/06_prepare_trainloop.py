import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## for the code to work, install torchvision
## $ python -m pip install --user -I --no-deps torchvision
import torchvision
from torchvision import datasets, transforms


def load_data(
    somepath,
    norm_loc=(0.1307,),  ## mu of normal dist to normalize by
    norm_scale=(0.3081,),  ## sigma of normal dist to normalize by
    train_kwargs={"batch_size": 64},
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
        train_kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True})
        test_kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": False})

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader


class MyNetwork(nn.Module):
    """a very basic relu neural network involving conv, dense, max_pool and dropout layers"""

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # x.shape = (batchsize, 10)
        output = F.log_softmax(x, dim=1)

        return output


def main(somepath="./pytorch-data"):
    """load the data set and run a random init CNN on it"""

    train_loader, test_loader = load_data(somepath)
    model = MyNetwork()

    optimizer = optim.Adadelta(model.parameters(), lr=1.e-3)
    max_nepochs = 1
    log_interval = 100

    init_params = list(model.parameters())[0].clone().detach()

    model.train(True)

    for epoch in range(1, max_nepochs + 1):

        for batch_idx, (X, y) in enumerate(train_loader):
            # print("train", batch_idx, X.shape, y.shape)

            optimizer.zero_grad()

            prediction = model(X)

            loss = F.nll_loss(prediction, y)

            loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch:",
                    epoch,
                    "Batch:",
                    batch_idx,
                    "Total samples processed",
                    (batch_idx + 1) * train_loader.batch_size,
                    "Loss:",
                    loss.item(),
                )

    final_params = list(model.parameters())[0].clone().detach()
    assert not torch.allclose(init_params, final_params)


if __name__ == "__main__":
    main()
    print("Ok. Checkpoint on training a model reached.")