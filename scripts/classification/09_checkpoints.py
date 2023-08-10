import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## for the code to work, install torchvision
## $ python -m pip install --user -I --no-deps torchvision
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
from tensorboardX import SummaryWriter


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
        # TODO where do the magic numbers come from?
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

    # is a GPU available?
    cuda_present = torch.cuda.is_available()
    ndevices = torch.cuda.device_count()
    use_cuda = cuda_present and ndevices > 0
    device = torch.device("cuda" if use_cuda else "cpu")  # "cuda:0" ... default device
    # "cuda:1" would be GPU index 1, "cuda:2" etc
    print("chosen device:", device, "use_cuda=", use_cuda)

    train_loader, test_loader = load_data(somepath, use_cuda=use_cuda)
    model = MyNetwork().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.e-3)
    max_nepochs = 1
    log_interval = 100

    init_params = list(model.parameters())[0].clone().detach()
    writer = SummaryWriter(log_dir="logs", comment="this is the test of SummaryWriter")
    model.train(True)

    chpfolder = Path("chkpts")
    if not chpfolder.is_dir():
        chpfolder.mkdir()

    for epoch in range(1, max_nepochs + 1):

        for batch_idx, (X, y) in enumerate(train_loader):
            # print("train", batch_idx, X.shape, y.shape)
            X, y = X.to(device), y.to(device)
            # download from GPU to CPU: X_cpu = X.cpu()
            # download from GPU to CPU: X_cpu = X.to(torch.device("cpu"))
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
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/train/batch10", loss.item(), batch_idx)

        # epoch finished
        cpath = chpfolder / f"epoch-{epoch:03.0f}.pth"
        torch.save(
            {
                "final_epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            cpath,
        )

        assert cpath.is_file() and cpath.stat().st_size > 0

    final_params = list(model.parameters())[0].clone().detach()
    assert not torch.allclose(init_params, final_params)

    # when to reload chkp, e.g. for doing inference
    payload = torch.load(cpath)
    model_from_ckpt = MyNetwork()
    model_from_ckpt.load_state_dict(payload['model_state_dict'])
    # continue learning/training after this
    loaded_params = list(model_from_ckpt.parameters())[0]
    assert torch.allclose(loaded_params, final_params)


if __name__ == "__main__":
    main()
    print("Ok. Checkpoint on training with checkpoints reached.")
