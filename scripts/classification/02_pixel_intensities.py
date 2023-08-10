import torch
import torchvision
from torchvision import datasets


def main(somepath="./pytorch-data", n_samples=32):

    raw_dataset = datasets.MNIST(somepath, download=True)

    if n_samples is None:
        n_samples = len(raw_dataset)

    imglist = []
    lbllist = []
    for i in range(n_samples):
        img, lbl = raw_dataset[i]
        imglist.append(img)
        lbllist.append(lbl)

    trf = torchvision.transforms.PILToTensor()
    timages = []
    for img in imglist:
        # print(type(img), img.size)
        timages.append(trf(img))
        # print(type(timages[-1]), timages[-1].shape)
        # print("____")

    # TODO: better torch.stack to keep channel dimensions
    print("torch.cat", torch.cat(timages).shape)
    print("torch.stack", torch.stack(timages).shape)

    batch = torch.stack(timages)
    assert batch.ndim == 4
    assert batch.shape == (n_samples, 1, 28, 28)
    print(batch.dtype)  # those are still integer valued images with range 0-255

    # print("batch mean value (raw) :", batch.mean())
    print("batch min value (fp32):", batch.float().min())
    print("batch max value (fp32):", batch.float().max())
    print("batch mean value (fp32):", batch.float().mean())
    print("batch std  value (fp32):", torch.std(batch.float()))

    # more information can be obtained from the API documentation:
    # https://pytorch.org/docs/stable/tensors.html
    nbatch = batch / 255.0
    print()
    print(nbatch.dtype)
    print("normalized batch min value (fp32):", nbatch.min())
    print("normalized batch max value (fp32):", nbatch.max())
    print("normalized batch mean value (fp32):", nbatch.mean())
    print("normalized batch std  value (fp32):", torch.std(nbatch))


if __name__ == "__main__":
    main()
