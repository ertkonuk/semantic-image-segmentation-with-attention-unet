# . . import libraries
import os
from pathlib import Path
# . . pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision

# . . numpy
import numpy as np
# . . scikit-learn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# . . matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as npimg
# . .  set this to be able to see the figure axis labels in a dark theme
from matplotlib import style
#style.use('dark_background')
# . . to see the available options
# print(plt.style.available)

from torchsummary import summary

# . . import libraries by tugrulkonuk
import utils
from dataset import TGSSaltDataset
from utils import parse_args
from model import *
from trainer import Trainer
from callbacks import ReturnBestModel, EarlyStopping

# . . parse the command-line arguments
args = parse_args()

# . . set the device
if torch.cuda.is_available():  
    device = torch.device("cuda")  
else:  
    device = torch.device("cpu")      

# . . set the default precision
dtype = torch.float32

# . . use cudnn backend for performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# . . parameters
# . . user-defined
num_epochs    = args.epochs
batch_size    = args.batch_size
learning_rate = args.lr
train_size    = args.train_size
min_delta     = args.min_delta
patience      = args.patience 
num_workers   = args.num_workers
pin_memory    = args.pin_memory
jprint        = args.jprint
# . . computed
test_size     = 1.0 - train_size

# . . data root path
dataroot = 'competition_data'
# . . path for training data
trainpath = dataroot+'/train'
# . . validation data
testpath = dataroot+'/test'

# . . get the list of files
allfiles = utils.get_file_list(trainpath+'/images')
allidx = np.linspace(0, len(allfiles)-1, len(allfiles), dtype=np.int)

train_idx, valid_idx = train_test_split(allidx, test_size=0.2, shuffle=True)
train_idx = np.unique(train_idx)
valid_idx = np.unique(valid_idx)


train_files = np.array(allfiles)[train_idx]
valid_files = np.array(allfiles)[valid_idx]


#train_files = utils.get_file_list(trainpath+'/images')
test_files = utils.get_file_list(testpath+'/images')

# . . read depths 
#depths = pd.read_csv(dataroot+'/depths.csv')

# . . the training set
train_dataset = TGSSaltDataset(trainpath, train_files, transform=None, mask=True, imgsize=(128,128), grayscale=True)

# . . the validation set
valid_dataset = TGSSaltDataset(trainpath, valid_files, transform=None, mask=True, imgsize=(128,128), grayscale=True)

# . . the test set
test_dataset = TGSSaltDataset(testpath, test_files, transform=None, mask=False, imgsize=(128,128), grayscale=True)

# . . the training loader: shuffle
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=pin_memory)

# . . the test loader: no shuffle
validloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)


# . . test the loader by retrieveing the batch info 
utils.batchinfo(trainloader, label=True)
utils.batchinfo(validloader, label=True)

# . . plot some images
images, masks = next(iter(trainloader))
utils.plot_masks(images, masks, 4, 4, figsize=(2,2))

# . . instantiate the model
model = AttentionUNet()

# . . send model to device (GPU)
model.to(device)

# . . show a summary of the model
summary(model, (1, 128, 128))

# . . create the trainer
trainer = Trainer(model, device)

# . . compile the trainer
# . . define the loss
criterion = nn.BCEWithLogitsLoss()

# . . define the optimizer
optimparams = {'lr':learning_rate
              }

# . . define the callbacks
cb=[ReturnBestModel(monitor='train_loss'), EarlyStopping(monitor='train_loss', min_delta=min_delta, patience=patience)]

trainer.compile(optimizer='adam', criterion=criterion, callbacks=cb, jprint=jprint, **optimparams)

# . . the learning-rate scheduler
schedulerparams = {'factor':0.5,
                   'patience':5,
                   'threshold':1e-5,
                   'cooldown':5,
                   'min_lr':1e-4,                
                   'verbose':True               
                  }
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, **schedulerparams)


# . . train the network
train_loss, valid_loss = trainer.fit(trainloader, validloader, scheduler=scheduler, num_epochs=num_epochs)


# . . plot the test and validation losses
plt.plot(train_loss)
plt.plot(valid_loss)
plt.legend(['train_loss', 'valid_loss'])

# . . test the trained network
# . . retrieve a new batch
#images, masks_true = next(iter(validloader))
images, masks_true = next(iter(trainloader))

# . . send images to device
images = images.to(device)
with torch.no_grad():
    # . . pass through the network
    outputs = trainer.model(images)

# . . send to CPU 
outputs    = outputs.cpu()

IoU = utils.get_iou_score(outputs, masks).numpy()

# . . convert images and masks to numpy
images     = images.cpu().numpy()
masks_true = masks_true.numpy()

masks_unet = torch.where((outputs)<0., torch.zeros_like(outputs), torch.ones_like(outputs)).numpy()

utils.compare_masks(images, masks_true, masks_unet, 4, 8, iou=None, figsize=(2,2))

#
# . . save the model
torch.save(trainer.model.state_dict(), 'models/final_model.pt')
