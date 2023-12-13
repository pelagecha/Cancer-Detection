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
	def __init__(self):
		super(Net, self).__init__()
		# Accept one input channel (image), outputs 20 matrices, 5x5 kernel	
		self.conv1 = nn.Conv2d(1, 20, 5)
		# Takes output of previous conv into input
		self.conv2 = nn.Conv2d(20, 50, 5)
		# Take output of conv to connected
		self.fc1 = nn.Linear(4*4*50, 500)
		# Repeat
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
def train(model, device, train_loader, optimizer, epoch):
	# Set to training mode (parameters can be changed)
	model.train()
	# Go over training data set
	# Get input and output (x and y)
	# Batch index for printing loss etc.
	for batch_idx, (data, target) in enumerate(train_loader):
		# Move data in target to device
		data, target = data.to(device), target.to(device)
		# Reset computed gradients to 0
		optimizer.zero_grad()
		# Run data through forward pass of network (get output in current step)
		output = model(data)
		# Calculates loss at current step
		loss = F.nll_loss(output, target)
		# Back propagation (how much each parameter contributed to loss)
		loss.backward()
		# Changes parameters depending on computed loss
		optimizer.step()
		# Prints every 100th batch
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))


# Loads testing data set
def test(model, device, test_loader):
	# Set model to evaluation mode (won't change parameters)
	model.eval()
	# Counts number of correct test sets
	correct = 0	
	# Test loss
	test_loss = 0
	# Speeds up progress as calculates no gradients
	with torch.no_grad():
		# iterates through test dat
		for data, target in test_loader:
			# Move data to device
			data, target = data.to(device), target.to(device)
			# Put data through model
			output = model(data)
			# Adds up all loss
			test_loss += F.nll_loss(output, target, reduction='sum').item()
			# Predicted output (gets index of maximum probability in output factor)
			pred = output.argmax(dim=1, keepdim=True)
			# Counts number of correct test sets
			correct += pred.eq(target.view_as(pred)).sum().item()

	# Finds loss average
	test_loss /= len(test_loader.dataset)
	
	print('\nTest set: Average loss: {: 4f}, Accuracy: {}/{} ({:.0f}\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100 * correct / len(test_loader)))

# Shuffles data, splits into batches
train_loader = torch.utils.data.DataLoader(
	# fetches training data
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