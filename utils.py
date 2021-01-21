# . . GPUtil for memory management
import platform
import numpy as np
import ntpath
import os, glob
import argparse
import torch
import matplotlib.pyplot as plt 

# . . do not import on MacOSX
if platform.system() is not 'Darwin':
    import GPUtil
# . . parse the command line parameters
def parse_args():

    parser = argparse.ArgumentParser(description='physics informed neural networks for 2D AWE solutions.')   
    # . . data directory
    default_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/'
    parser.add_argument('--datapath',     type=str,   default=default_dir, help='data path')
    # . . hyperparameters
    parser.add_argument('--lr',           type=float, default=1e-3,  help='learning rate')
    parser.add_argument('--batch_size',   type=int,   default=64,    help='training batch size')
    parser.add_argument('--train_size',   type=float, default=0.8,   help='fraction of grid points to train')

    # . . training parameters
    parser.add_argument('--epochs',       type=int,   default=100,   help='number of epochs to train')

    # . . parameters for early stopping
    parser.add_argument('--patience',     type=int,   default=10,      help='number of epochs to wait for improvement')
    parser.add_argument('--min_delta',    type=float, default=0.0005,  help='min loss function reduction to consider as improvement')
    
    # . . parameters for data loaders
    parser.add_argument('--num_workers',    type=int,  default=8,      help='number of workers to use inf data loader')
    parser.add_argument('--pin_memory' ,    type=bool, default=False,  help='use pin memory for faster cpu-to-gpu transfer')

    parser.add_argument('--jprint',       type=int,   default=1,   help='print interval')

    # . . parse the arguments
    args = parser.parse_args()

    return args


def gpuinfo(msg=""):
    print("------------------")
    print(msg)
    print("------------------")
    GPUtil.showUtilization()
    print("------------------")

def devinfo(device):
    print("------------------")
    print("torch.device: ", device)
    print("------------------")
   
def batchinfo(loader, label=True):

    print("------------------")
    print("There are {} batches in the dataset".format(len(loader)))
    if label:
        for x, y in loader:
            print("For one iteration (batch), there are:")
            print("Data:    {}".format(x.shape))
            print("Label:   {}".format(y.shape))
            break   
    else:
        for x in loader:
            print("For one iteration (batch), there are:")
            print("Data:    {}".format(x.shape))
            break  
    print("------------------")

def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    taken from: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(padding) is not tuple:
        padding = (padding, padding)
    
    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

# . . strips the path name off from a full path 
def strip_path(path):
    head, tail = ntpath.split(path)
    return tail    

# . . get the list of files w/ specified file ext in a directory  w/out ext
def get_file_list(path, ext='png'):
    # . . all files in the path
    files=glob.glob(path + '/*.' + ext)    
    
    # . . strip the file etension
    flist = []
    for i in range(len(files)):
        # . . strip the path
        _, tail = ntpath.split(files[i])
        flist.append(tail.rsplit( ".", 1 )[ 0 ])
    
    return flist

# . . plot the seismic with mask
def plot_masks(images, masks, nrows, ncols, alpha=0.25, figsize=(5,5)):

    # . . adjust figsize
    w, h = figsize
    w *= ncols
    h *= nrows
    figsize = (w,h)

    # . .number of figures
    num_figs = nrows*ncols

    # . . the number of images
    num_images = len(images)
    # . . generate random indexes
    rand_idx = np.random.randint(0, num_images, num_figs) #generate random indexes

    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=figsize)

    idfig = 0
    for j in range(nrows):
        for i in range(ncols):    
            ax[j,i].imshow(images[idfig,:,:,:].squeeze(), cmap='gray')
            ax[j,i].imshow( masks[idfig,:,:,:].squeeze(), cmap='Greens', alpha=alpha)
            ax[j,i].contour(masks[idfig,:,:,:].squeeze(), colors='k', levels=[0.5])
            ax[j,i].axes.get_xaxis().set_visible(False)
            ax[j,i].axes.get_yaxis().set_visible(False)

            idfig += 1

    fig.tight_layout()
    plt.show()

# . . plot the seismic with mask
def compare_masks(images, masks_true, masks_pred, nrows, ncols, iou=None, alpha=0.25, figsize=(5,5)):

    # . . adjust figsize
    w, h = figsize
    w *= ncols
    h *= nrows
    figsize = (w,h)

    # . .number of figures
    num_figs = nrows*ncols

    # . . the number of images
    num_images = len(images)
    # . . generate random indexes
    rand_idx = np.random.randint(0, num_images, num_figs) #generate random indexes

    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=figsize)

    idfig = 0
    for j in range(nrows):
        for i in range(ncols):    
            ax[j,i].imshow(images[idfig,:,:,:].squeeze(), cmap='gray')
            ax[j,i].imshow( masks_true[idfig,:,:,:].squeeze(), cmap='Greens', alpha=alpha)
            ax[j,i].contour(masks_true[idfig,:,:,:].squeeze(), colors='k', levels=[0.5])
            ax[j,i].imshow( masks_pred[idfig,:,:,:].squeeze(), cmap='OrRd', alpha=alpha)
            ax[j,i].contour(masks_pred[idfig,:,:,:].squeeze(), colors='r', levels=[0.5])
            if iou is not None:
                ax[j,i].set_title("IoU: " + str(round(iou[idfig], 2)), loc = 'left')
            ax[j,i].axes.get_xaxis().set_visible(False)
            ax[j,i].axes.get_yaxis().set_visible(False)

            idfig += 1
    plt.suptitle("Green: salt, Red: prediction")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

# . . compute the intersection over union
# . . torch
def get_iou_score(outputs, labels):

    A = labels.squeeze().bool()
    pred = torch.where(outputs<0., torch.zeros_like(outputs), torch.ones_like(outputs))
    B = pred.squeeze().bool()
    intersection = (A & B).float().sum((1,2))
    union = (A| B).float().sum((1, 2)) 
    iou = (intersection + 1e-6) / (union + 1e-6)  
    return iou
   