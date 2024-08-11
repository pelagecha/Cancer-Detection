"Data" folder contains scans of lesions. Types 1~8 correspond to bone, abdomen, mediastinum, liver, lung,
kidney, soft tissue, and pelvis, respectively.

Typical Pattern in CNNs:
Convolution -> Activation -> Convolution -> Activation -> Pooling
Convolution -> Activation -> Convolution -> Activation -> Pooling
Flatten -> Fully Connected -> Activation
Fully Connected -> Activation -> Output Layer

Universal function approximator

-   assigns label / category / class

Loss function (Supervised Learning)

-   measures how wrong compared to ground truth (lower = better)
-   convex

Optimiser

-   lowers loss

Back Propagation

-   propagates error backwards through network
-   can calculate amount each parameter contributed

Stochastic Gradient Descent

-   'down the gradient of slope' so error becomes lower

Activation function (non-linear function)

-   applies to every neuron output
-   introduces non-linearity

Linear function

-   Additivity: f(x+y) = f(x) + f(y)
-   Homogeneity: f(ax) = af(x)

Hyperparameters

-   parameters to change for optimisation
-   -   learning rate (how much parameters change by)
-   -   momentum (follow (repeat) changes in a parameter)
-   -   # layers, neurons
-   -   weight initialisation
-   -   batch size (operations done in batches)

Split data to aovid overfitting (memorising not learning)

    MODEL DEFINITION

import torch
// Activation functions, layer definitions, loss functions
import torch.nn as nn
// Same as 'nn' but all functional
import torch.nn.functional as F
// Optimisers
import torch.optim as optim
// dataset definitions, image transformation tools
from torchvision import datasets, transforms

// Inherits from nn.module
class Net(nn.module):
def **init**(self):
// fully connected layers, inputs, outputs
// contains 3 layers: 500, 1000, 10
self.fc1 = nn.Linear(784, 500)
self.fc2 = nn.Linear(500, 1000)
self.fc3 = nn.Linear(1000, 10)

    // backwards paths calculated automatically
    // x = multidimensional array
    def forward(self, x):
    	// Resize array so second dimension is 784
    	x = x.view(-1, 784)
    	// Apply 'relu' activation function to output of layers 1 and 2
    	x = F.relu(self.fc1(x))
    	x = F.relu(self.fc2(x))
    	// Apply linear, then logarithmic softmax function (only want highest probability)
    	x = self.fc3(x)
    	return F.log_softmax(x, dim=1)

    LOAD DATA

// Shuffles data, splits into batches
train_loader = torch.utils.data.DataLoader(
// fetches training data
datasets.MNIST('../data', train=True, download=True
transform=transforms.Compose([
transforms.ToTensor()
transforms.Normalize((0, 1307), (0.3081))
])),
batch_size=128, shuffle=True)

// Fetches testing data
test_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=False, transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=1000, shuffle=True)

    TRAINING

// model, device (GPU, CPU), training data set, initialised object representing optimizer,
// epoch (entire data set gone over in one epoch)
def train(model, device, train_loader, optimizer, epoch):
// Set to training mode (parameters can be changed)
model.train()
// Go over training data set
// Get input and output (x and y)
// Batch index for printing loss etc.
for batch_idx, (data, target) in enumerate(train_loader):
// Move data in target to device
data, target = data.to(device), target.to(device)
// Reset computed gradients to 0
optimizer.zero_grad()
// Run data through forward pass of network (get output in current step)
output = mode(data)
// Calculates loss at current step
loss = F.nll_loss(output, target)
// Back propagation (how much each parameter contributed to loss)
loss.backward()
// Changes parameters depending on computed loss
optimizer.step()
// Prints every 100th batch
if batch_idx % 100 == 0:
print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
epoch, batch_idx _ len(data), len(train_loader.dataset), 100. _ batch_idx / len(train_loader), loss.item()))

    TESTING

// Loads testing data set
def test(model, device, test_loader):
// Set model to evaluation mode (won't change parameters)
model.eval()
// Counts number of correct test sets
correct = 0
// Test loss
test_loss = 0
// Speeds up progress as calculates no gradients
with torch.no_grad();
// iterates through test dat
for data, target in test_loader:
// Move data to device
data, target = data.to(device), target.to(device)
// Put data through model
output = model(data)
// Adds up all loss
test_loss += F.nll_loss(output, target, reduction='sum').item()
// Predicted output (gets index of maximum probability in output factor)
pred = output.argmax(dim=1, keepdim=True)
// Counts number of correct test sets
correct += pred.eq(target.view_as(pred)).sum().item()

    // Finds loss average
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {: 4f}, Accuracy: {}/{} ({:.0f}\n'.format(
    	test_loss, correct, len(test_loader.dataset),
    	100 * correct / len(test_loader_dataset)))

    PARAMETER TUNING AND TRAINING

// See if cuda available or GPU
use_cuda = torch.cuda.is_available()

// Set manual random seed
torch.manual_seed(42)

// Use cuda or CPU as device (can be "cuda:1" to use diff card)
device = torch.device("cuda" if use_cuda else "cpu")

// Initialise network and move to device
model = Net().to(device)

// Initialises optimiser at stochastic gradient descsent, learning rate
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

// Runs test before training
test(model, device, test_loader)

// Train and test network for some alternating epoch (look into this more)
for epoch in range(1, 3 + 1):
train(model, device, train_loader, optimizer, epoch)
test(model, device, test_loader)

    CONVOLUTION

Apply matrix (convolution kernel) for edge detection

Pixel considered in combination with neighbours

Difference - use features to classify, not raw pixel data

More resilient against scaling, rotation and repositioning

How they work

-   convolutional kernel (matrix)
-   slide matrix over image
-   perform simple element-wise multiplication and sumation
-   kernels need to be learned
-   hyperparameters
-   -   padding (pad input image before applying convolution)
-   -   stride (how much kernel moved)

Pooling

-   reduce dimensionality (precise location not needed)
-   maximum pooling (takes maximum values from outputs)
-   average pooling (takes average from sub-matrices)

Learned features

-   common textures and shapes found in training set
-   larger layer representation produces more meaningful features

    DIFFERENCE BETWEEN CONVOLUTION

// Accept one input channel (image), outputs 20 matrices, 5x5 kernel
self.conv1 = nn.Conv2d(1, 20, 5)
// Takes output of previous conv into input
self.conv2 = nn.Conv2d(20, 50, 5)
// Take output of conv to connected
self.fc1 = nn.Linear(4*4*50, 500)
// Repeat
self.fc2 = nn.Linear(500, 10)

x = F.relu(self.conv1(x))
// Apply maximum pooling (size 2, stride 2)
x = F.max_pool2d(x, 2, 2)
x = F.relu(self.conv2(x))
x = F.max_pool2d(x, 2, 2)
// Transform output of pooling and convolution to 1d
x = x.view(-1, 4*4*50)
x = F.relu(self.fc1(x))
x = self(fc2(x))
return F.log_softmax(x, dim=1)

    PRETRAINED MODELS

Transfer learning - use other AI's learning as starting point

Object detection provides object position (bounding box)

n DL_info.csv, each row is the information of a lesion in DeepLesion. The meaning of the columns
are:

1. File name. Please replace the last underscore with / or \ to indicate sub-folders.
2. Patient index starting from 1.
3. Study index for each patient starting from 1. There are 1~26 studies for each patient.
4. Series ID.
5. Slice index of the key slice containing the lesion annotation, starting from 1.
6. 8D vector, the image coordinates (in pixel) of the two RECIST diameters of the lesion. [x11,
   y11, x12, y12, x21, y21, x22, y22]. The first 4 coordinates are for the long axis. Please see our paper
   and its supplementary material for further explanation.
7. 4D vector, the bounding-box [x1, y1, x2, y2] of the lesion (in pixel) estimated from the RECIST
   diameters, see our paper.
8. 2D vector, the lengths of the long and short axes. The unit is pixels.
9. The relative body position of the center of the lesion. The z-coordinates were predicted by the
   self-supervised body part regressor. See our paper for details. The coordinates are approximate
   and just for reference.
10. The type of the lesion. Types 1~8 correspond to bone, abdomen, mediastinum, liver, lung,
    kidney, soft tissue, and pelvis, respectively. See our paper for details. The lesion types are
    coarsely defined and just for reference. Only the lesions in the val and test sets were annotated
    with others denoted as -1.
11. This field is set to 1 if the annotation of this lesion is possibly noisy according to manual check.
    We found 35 noisy annotations out of 32,735 till now.
12. Slice range. Context slices neighboring to the key slice were provided in this dataset. For
    example, in the first lesion, the key slice is 109 and the slice range is 103~115, meaning that
    slices 103~115 are provided. For most lesions, we provide 30mm extra slices above and below
    the key slice, unless the long axis of the lesion is larger than this thickness (then we provide
    more) or the beginning or end of the volume is reached.
13. Spacing (mm per pixel) of the x, y, and z axes. The 3rd value is the slice interval, or the physical
    distance between two slices.
14. Image size.
15. The windowing (min~max) in Hounsfield unit extracted from the original DICOM file.
16. Patient gender. F for female and M for male.
17. Patient age.
18. Official randomly generated patient-level data split, train=1, validation=2, test=3

import torch

# Activation functions, layer definitions, loss functions

import torch.nn as nn

# Same as 'nn' but all functional

import torch.nn.functional as F

# Optimisers

import torch.optim as optim

# dataset definitions, image transformation tools

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Inherits from nn.module

class Net(nn.Module):
def **init**(self):
super(Net, self).**init**() # Accept one input channel (image), outputs 20 matrices, 5x5 kernel
self.conv1 = nn.Conv2d(1, 20, 5) # Takes output of previous conv into input
self.conv2 = nn.Conv2d(20, 50, 5) # Take output of conv to connected
self.fc1 = nn.Linear(4*4*50, 500) # Repeat
self.fc2 = nn.Linear(500, 10)

# backwards paths calculated automatically

# x = multidimensional array

def forward(self, x):
x = F.relu(self.conv1(x))

    # Apply maximum pooling (size 2, stride 2)
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)

    # Transform output of pooling and convolution to 1d
    x = x.view(-1, 4*4*50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return F.log_softmax(x, dim=1)

# model, device (GPU, CPU), training data set, initialised object representing optimizer,

# epoch (entire data set gone over in one epoch)

def train(model, device, train_loader, optimizer, epoch): # Set to training mode (parameters can be changed)
model.train() # Go over training data set # Get input and output (x and y) # Batch index for printing loss etc.
for batch_idx, (data, target) in enumerate(train_loader): # Move data in target to device
data, target = data.to(device), target.to(device) # Reset computed gradients to 0
optimizer.zero_grad() # Run data through forward pass of network (get output in current step)
output = model(data) # Calculates loss at current step
loss = F.nll_loss(output, target) # Back propagation (how much each parameter contributed to loss)
loss.backward() # Changes parameters depending on computed loss
optimizer.step() # Prints every 100th batch
if batch_idx % 100 == 0:
print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
epoch, batch_idx _ len(data), len(train_loader.dataset), 100. _ batch_idx / len(train_loader), loss.item()))

# Loads testing data set

def test(model, device, test_loader): # Set model to evaluation mode (won't change parameters)
model.eval() # Counts number of correct test sets
correct = 0 # Test loss
test_loss = 0 # Speeds up progress as calculates no gradients
with torch.no_grad(): # iterates through test dat
for data, target in test_loader: # Move data to device
data, target = data.to(device), target.to(device) # Put data through model
output = model(data) # Adds up all loss
test_loss += F.nll_loss(output, target, reduction='sum').item() # Predicted output (gets index of maximum probability in output factor)
pred = output.argmax(dim=1, keepdim=True) # Counts number of correct test sets
correct += pred.eq(target.view_as(pred)).sum().item()

    # Finds loss average
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {: 4f}, Accuracy: {}/{} ({:.0f}\n'.format(
    	test_loss, correct, len(test_loader.dataset),
    	100 * correct / len(test_loader)))

# Shuffles data, splits into batches

train_loader = torch.utils.data.DataLoader( # fetches training data
datasets.MNIST('../data', train=True, download=True,
transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0, 1307), (0.3081,))
])),
batch_size=128, shuffle=True)

# Fetches testing data

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=False, transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=1000, shuffle=True)

# See if cuda available or GPU

use_cuda = torch.cuda.is_available()

# Set manual random seed

torch.manual_seed(42)

# Use cuda or CPU as device (can be "cuda:1" to use diff card)

device = torch.device("cuda" if use_cuda else "cpu")

# Initialise network and move to device

model = Net().to(device)

# Initialises optimiser at stochastic gradient descsent, learning rate

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

# Runs test before training

test(model, device, test_loader)

# Train and test network for some alternating epoch (look into this more)

for epoch in range(1, 3 + 1):
train(model, device, train_loader, optimizer, epoch)
test(model, device, test_loader)
.
