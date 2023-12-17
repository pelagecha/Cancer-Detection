from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

data_dir = "binary"
# torch.set_num_threads(4)


# README available at: https://nihcc.app.box.com/v/DeepLesion/file/306056134060
# labeled data at: https://nihcc.app.box.com/v/ChestXray-NIHCC
# load images as torch tensors from a folder for training
# each tensor is shaped 3xWIDTHxHEIGHT (3 for RGB) |img, label = dataset[0] print(img.shape, label)|
train_ds = ImageFolder(data_dir + "/train", transform=ToTensor())
val_ds = ImageFolder(data_dir + "/validate", transform=ToTensor())

'''SPLITTING THE DATASET INTO TEST AND VALIDATE'''
# random_seed = 42 # could be any number
# torch.manual_seed(random_seed)
# val_size = 5000 # validation set size
# train_size = len(dataset) - val_size
#
# train_ds, val_ds = random_split(dataset, [train_size, val_size])
# print(len(train_ds), len(val_ds))


'''DATA LOADING'''
batch_size = 64  # can be changed (doubled)
# shuffling leads to faster training, num_workers specifies number of cpu cores used, pin_memory if images are same size.
train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size * 2, shuffle=True, pin_memory=True)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# simple_model = nn.Sequential(
#     nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2)
# )

class Simple(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 8 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 16 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32 x 8 x 8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, xb):
        return self.network(xb)




# num of channels(RGB), num of kernels, kernel size, stepsize, padding


'''MAIN MODEL'''


class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)


'''TRAINING'''


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    count = 0
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            count += 1
            print(count)
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


'''HELPER FUNCTIONS'''


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.title('Accuracy vs epoch')
    plt.show()


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('val_loss') for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs epoch')
    plt.show()


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        if not torch.backends.mps.is_available():
            return torch.device('cpu')
        else:
            return torch.device("mps")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(labels))


def show_batch(dl):
    """
    show all images in the dataset as a grid
    """
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))  # Adjust nrow as needed
        plt.show()
        break


# def show_example(img, label):
#     print("Label: ", dataset.classes[label], "(" + str(label) + ")")
#     plt.imshow(img.permute(1, 2, 0)) # converts 3x32x32 into 32x32x3
#     plt.show()


def apply_kernel(image, kernel):
    """
    KERNEL APPLICATION (CONVERSTION TO VECTOR)
    """
    ri, ci = image.shape  # image dimensions
    rk, ck = kernel.shape  # kernel dimensions
    ro, co = ri - rk + 1, ci - ck + 1
    output = torch.zeros([ro, co])
    for i in range(ro):
        for j in range(co):
            output[i, j] = torch.sum(image[i:i + rk, j:j + ck] * kernel)
        return output


if __name__ == '__main__':
    device = get_default_device()
    val_dl = DeviceDataLoader(val_dl, device)
    train_dl = DeviceDataLoader(train_dl, device)
    model = to_device(Simple(), device)
    # print(device)
    # print(evaluate(model, val_dl))
    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.0005
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    print(history)

    plot_accuracies(history)
    # plot_losses(history)
    # to_device(model, device)
    # show_batch(val_dl)
    # show_example(*dataset[1]) # unpacks the tuple inline

'''
Training set is used to train the model. It's like memorizing the testbook.
Validation set is used to test the model while it's being trained, and choose the best one. It's like doing the exam.
Test set is used to test the model
'''
