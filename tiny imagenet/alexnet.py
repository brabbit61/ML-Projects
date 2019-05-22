import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary

datadir = '/home/jenit/Desktop/work/tiny imagenet/tiny-imagenet-200/'
traindir = datadir + 'train/'
validdir = datadir + 'val/'
batch_size = 128
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)

model = models.alexnet(pretrained=True)
for param in model.parameters():
	param.requires_grad = False

model.classifier[6] = nn.Sequential( nn.Linear(4096, 200), nn.LogSoftmax(dim=1))

summary(model, input_size=(3, 224, 224), batch_size=batch_size)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())
n_epochs = 10

train_loader = dataloaders['train']
val_loader = dataloaders['val']
for epoch in range(n_epochs):
	print(epoch)
	train_loss = 0.0
	valid_loss = 0.0
	train_acc = 0
	valid_acc = 0
	model.train()
	for ii, (data, target) in enumerate(train_loader):
		print(ii)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * data.size(0)
		_, pred = torch.max(output, dim=1)
		correct_tensor = pred.eq(target.data.view_as(pred))
		accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
		train_acc += accuracy.item() * data.size(0)
	
	model.epochs += 1

	with torch.no_grad():
		model.eval()
		for data, target in val_loader:
			output = model(data)
			loss = criterion(output, target)
			valid_loss += loss.item() * data.size(0)
			_, pred = torch.max(output, dim=1)
			correct_tensor = pred.eq(target.data.view_as(pred))
			accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
			valid_acc += accuracy.item() * data.size(0)
		
		train_loss = train_loss / len(train_loader.dataset)
		valid_loss = valid_loss / len(val_loader.dataset)
		train_acc = train_acc / len(train_loader.dataset)
		valid_acc = valid_acc / len(val_loader.dataset)

		print("\nEpoch: "+str(epoch)+" Training Loss:"+str(train_loss)+" Validation Loss:"+str(valid_loss))
		print("Training Accuracy:"+ str(100 * train_acc)+"% Validation Accuracy:"+ str(100 * valid_acc)+"%\n\n")

model.optimizer = optimizer
'''
checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
}
checkpoint['classifier'] = model.classifier
checkpoint['state_dict'] = model.state_dict()
checkpoint['optimizer'] = model.optimizer
checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
torch.save(checkpoint, "weights/")
'''
