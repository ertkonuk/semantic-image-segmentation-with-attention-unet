import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# . .                . . #
# . . Attention UNet . . #
# . .                . . #

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x, out_att=False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        if out_att:
            return x*psi, psi
        else:
            return x*psi

class AttentionUNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttentionUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16)
        self.Conv2 = conv_block(ch_in=16,ch_out=32)
        self.Conv3 = conv_block(ch_in=32,ch_out=64)
        self.Conv4 = conv_block(ch_in=64,ch_out=128)
        self.Conv5 = conv_block(ch_in=128,ch_out=256)

        self.Up5      = up_conv(ch_in=256,ch_out=128)
        self.Att5     = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4      = up_conv(ch_in=128,ch_out=64)
        self.Att4     = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Att3 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        
        self.Up2 = up_conv(ch_in=32,ch_out=16)
        self.Att2 = Attention_block(F_g=16,F_l=16,F_int=8)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        
        d4 = self.Att5(d5,x4)        
        d5 = torch.cat((d4,d5),dim=1)                
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(d4,x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3,x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2,x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def forward_with_attention(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        
        d4, att5 = self.Att5(d5,x4, out_att=True) 
        d5 = torch.cat((d4,d5),dim=1)                
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3, att4 = self.Att4(d4,x3, out_att=True)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2, att3 = self.Att3(d3,x2, out_att=True)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1, att2 = self.Att2(d2,x1, out_att=True)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, [att5, att4, att3, att2]



# . .               . . #
# . . Parabolic CNN . . #
# . .               . . #
class PackNet(nn.Module):

    def __init__(self, h, NG, L_mode='zero', device='cuda'):
        super().__init__()

        _, self.depth = NG.shape
        self.NG = NG
        self.h = h
        self.L_mode = L_mode
        self.device = device
        # . . added by tugrulkonuk
        self.num_classes = NG[0,0]
        self.L_mode = L_mode

        # . . CNN weights
        K, L = self._init_weights()
        # . . weights for the final layer: classifier
        W = torch.rand(self.num_classes, self.NG[-1, -1], 1, 1)*1e-3

        # . . send weights to device
        W = W.cuda()
        K = [Ki.cuda() for Ki in K]
        L = [Li.cuda() for Li in L]
        
        # . . register K and L as parameters: CNN weights
        #self.K = [nn.Parameter(Ki) for Ki in K]
        #self.L = [nn.Parameter(Li) for Li in L]
        self.K = nn.ParameterList([nn.Parameter(Ki) for Ki in K])
        self.L = nn.ParameterList([nn.Parameter(Li) for Li in L])
        # . .register the weights for the final classifier
        self.W = nn.Parameter(W)

    def _init_weights(self):
        nsteps = self.NG.shape[1]
        K = []
        L = []
        self.BN = []

        if self.L_mode=='laplacian':
            lap_stencil = torch.Tensor([[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]])/6
            lap_stencil.unsqueeze_(0).unsqueeze_(0)

        for i in range(nsteps):  
            Ki  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)*1e-3
            # Ki  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)

            if self.L_mode=='rand':
                Li  = torch.rand(np.asscalar(self.NG[1,i]), 1, 3, 3)*1e0
                # Li  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)*1e-3
            elif self.L_mode=='laplacian':
                Li = lap_stencil.repeat(self.NG[1,i], self.NG[0,i], 1, 1)*1e-2
            elif self.L_mode=='zero':
                Li  = torch.zeros(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)
            else:
                raise NotImplementedError()

            Ki   = projectTorchTensor(Ki)

            bni = nn.BatchNorm2d(self.NG[1,i])
            
            bni = bni.to(self.device)
            self.BN.append(bni)
            K.append(Ki)
            L.append(Li)

        return K, L


    def init_weights(self):
        nsteps = self.NG.shape[1]
        self.L_mode = L_mode
        K = []
        L = []
        self.BN = []

        if L_mode=='laplacian':
            lap_stencil = torch.Tensor([[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]])/6
            lap_stencil.unsqueeze_(0).unsqueeze_(0)

        for i in range(nsteps):  
            Ki  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)*1e-3
            # Ki  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)

            if L_mode=='rand':
                Li  = torch.rand(np.asscalar(self.NG[1,i]), 1, 3, 3)*1e0
                # Li  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)*1e-3
            elif L_mode=='laplacian':
                Li = lap_stencil.repeat(self.NG[1,i], self.NG[0,i], 1, 1)*1e-2
            elif L_mode=='zero':
                Li  = torch.zeros(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)
            else:
                raise NotImplementedError()

            Ki   = projectTorchTensor(Ki)

            bni = nn.BatchNorm2d(self.NG[1,i])

            if self.use_gpu:
                bni = bni.cuda()
            self.BN.append(bni)
            K.append(Ki)
            L.append(Li)
        
        return K, L
    
    def num_params(self, kernel_size=3):
        explicit_params = (np.prod(self.NG, axis = 0)*kernel_size**2).sum()
        
        implicit_params = 0
        layers = self.NG[0,:]
        _, opening_layers = np.unique(layers, return_index=True)
        for i, nchannels in enumerate(layers):
            if i not in opening_layers:
                implicit_params += nchannels*(kernel_size**2)

        return explicit_params, implicit_params

    def forward(self,x, enorm=False):
    
        nt = len(self.K)
        # . . previous time step
        xm = x*0

        if enorm:
            # . . get hte size of the input batch
            N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            E_norm = torch.zeros((N,H,W)).to(self.device)

        # . . time stepping
        for j in range(nt):
            batchnorm = self.BN[j]

            # . . if same width
            if self.NG[0,j] == self.NG[1,j]: 

                # Zero kernel means
                Kj = projectTorchTensor(self.K[j])
                #Lj = projectTorchTensor(self.L[j])

                # . . the explicit step
                z = conv3x3(x,Kj)
                z = batchnorm(z)
                z = F.relu(z)        
                z = conv3x3T(z, Kj)

                # .  . 2nd order
                xp  = 2.0*x - xm + self.h*self.h*z 
                ## .  . diffusion reaction
                #x  = x + self.h*z 
                
                ## . . the implicit Step
                #if not self.L_mode == 'zero':
                #    xp = batchnorm(xp)
                #    xp  = diagImpConvFFT(xp, Lj, self.h, L_mode=self.L_mode)

                if enorm:
                    # . . the energy norm
                    # . . kinetic energy
                    E_kin = (1./(2.0*self.h)) * torch.sum((xp - xm), dim=1).squeeze()
                    # . . potential energy
                    E_pot = torch.sum(z*z, dim=1).squeeze()
                    # . .the total ebergy of the layer/block
                    E_norm += E_kin + E_pot

                # . .  keep the previous layers
                xm = x
                x = xp                

            # . . change the number of channels/resolution
            else:
                z1 = conv3x3(x, self.K[j])

                z1 = batchnorm(z1)
                x  = F.relu(z1)

        # . . the final classifier
        x = conv1x1(x,self.W) 

        if enorm:
            return x, E_norm
        else:
            return x

# . . utility functions . . #
def projectTorchTensor(K): 
    n = K.data.shape
    K.data  = K.data.view(-1,9)
    M       = K.data.mean(1)
    for i in range(9):
        K.data[:,i] -= M
        
    K.data  = K.data.view(n[0],n[1],n[2],n[3])
    return K        

def conv3x3(x,K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=1)

def conv3x3T(x,K):
    """3x3 convolution transpose with padding"""
    #K = torch.transpose(K,0,1)
    return F.conv_transpose2d(x, K, stride=1, padding=1)

def conv1x1(x,K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=0)

def diagImpConvFFT(x, K, h, L_mode=None):
    """convolution using FFT of (I + h*K'*K)^{-1}"""
    n = x.shape
    m = K.shape
    mid1 = (m[2] - 1) // 2
    mid2 = (m[3] - 1) // 2
    Kp = torch.zeros(m[0],n[2], n[3],device=K.device)
    # this code also flips up-down and left-right
    Kp[:, 0:mid1 + 1, 0:mid2 + 1] = K[:, 0, mid1:, mid2:]
    Kp[:, -mid1:, 0:mid2 + 1] = K[:, 0, 0:mid1, -(mid2 + 1):]
    Kp[:, 0:mid1 + 1, -mid2:] = K[:, 0, -(mid1 + 1):, 0:mid2]
    Kp[:, -mid1:, -mid2:] = K[:, 0, 0:mid1, 0:mid2]

    xh = torch.rfft(x, 2, onesided=False)
    Kh = torch.rfft(Kp, 2, onesided=False)
    xKh = torch.zeros(n[0],n[1],n[2], n[3], 2,device=K.device)

    # dont need semi pos def if L is laplacian
    if L_mode == 'laplacian':
        t = 1.0/(h * torch.abs(Kh[:, :, :, 0]) + 1.0) 
    else:
        t = 1.0/(h * (Kh[:, :, :, 0] ** 2 + Kh[:, :, :, 1] ** 2) + 1.0)

    for i in range(n[0]):
        xKh[i, :,:, :, 0] = xh[i, :, :, :, 0]*t
        xKh[i, :,:, :, 1] = xh[i, :, :, :, 1]*t
    xK = torch.irfft(xKh, 2, onesided=False)
    return xK    


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    def plottable(x):
        return np.moveaxis(x.cpu().detach().numpy(), 1, -1).squeeze()

    h = 1e-1
    #NG = [ 3,3,3,3,3,3,3,3,3,
    #       3,3,3,3,3,3,3,3,3]
    # NG = [  3, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    #        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    NG = [ 3,  8, 16, 32,
           8, 16, 32, 3 ]
    NG = np.reshape(NG, (2,-1))
    
    net = PackNet(h, NG, False)
    K,L = net.init_weights(L_mode='laplacian')

    x = torch.zeros(1,3,32,32)
    # x = torch.zeros(1,3,128,128)
    x[:,:,8, 8] = 100
    plt.imshow(plottable(x))
    plt.show()

    X = net(x, K, L)
    print('lenK:',len(K))
    print('K0:',K[0].shape)
    print('K1:',K[1].shape)
    print('K2:',K[2].shape)
    print('K3:',K[3].shape)
    raise Exception(net, net.depth)
