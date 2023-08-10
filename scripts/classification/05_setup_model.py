import torch
## for the code to work, install torchvision
## $ python -m pip install --user -I --no-deps torchvision
import torchvision
from torchvision import datasets, transforms

## NB: in case torchvision cannot be found inside a jupyter notebook, fix the PYTHONPATH through
##     import sys
##     sys.path.append("/home/haicore-project-ws-hip-2021/mk7540/.local/lib/python3.8/site-packages/")


def load_data(
    somepath,
    norm_loc=(0.1307,),  ## mu of normal dist to normalize by
    norm_scale=(0.3081,),  ## sigma of normal dist to normalize by
    train_kwargs={"batch_size": 64},
    test_kwargs={"batch_size": 1_000},
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


import torch.nn as nn
import torch.nn.functional as F


class MyNetwork(nn.Module):
    """a very basic relu neural network involving conv, dense, max_pool and dropout layers"""

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # NOTE: 9216 comes from flattening the feature maps that are the
        # result from maxpooling the last conv layers output: 64 x 12 x 12
        # 64 * 12 * 12 = 9216
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        print("Shape after conv1", x.shape)

        x = self.conv2(x)
        x = F.relu(x)
        print("Shape after conv2", x.shape)

        x = F.max_pool2d(x, 2)
        print("Shape after max pool", x.shape)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        print("Shape after flattening", x.shape)

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

    batch_image = None
    batch_label = None

    for batch_idx, (X, y) in enumerate(train_loader):
        print("train", batch_idx, X.shape, y.shape)

        batch_image = X
        batch_label = y

        break

    model = MyNetwork()

    output = model(batch_image)
    assert len(output.shape) > 1
    assert output.shape[0] == batch_label.shape[0]

    prediction = output.argmax(dim=1)
    assert prediction.shape == batch_label.shape


if __name__ == "__main__":
    main()
    print("Ok. Checkpoint on model creation reached.")
