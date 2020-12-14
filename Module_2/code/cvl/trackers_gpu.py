import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import torch
import torch.fft
import torchvision.models as models
import cv2
from .image_io import crop_patch
from .lib import normalize, smooth_edge, get_2d_gauss, transf2ori
from .features import alexnetFeatures


class DCFMOSSETracker:
    def __init__(self, 
                 dev,
                 features = "alexnet",
                 lambda_ = 1e-5,
                 learning_rate = 1e-3,
                 sigma = 2):
        self.device = dev
        self. features_extractor = features
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.sig = sigma

        
        if self.features_extractor == "alexnet":
            self.model = alexnetFeatures(pretrained=True, progress = False).to(self.device)
        elif self.features_extractor == "vgg16":
            self.model = models.vgg16(pretrained=True).features[:12].to(self.device) # best layers -> 5
        elif self.features_extractor == "mobilenet":
            self.model = models.mobilenet_v2(pretrained=True).features[:6].to(self.device) # best layer -> 6 
        elif self.features_extractor == "resnet":
            resnet = models.resnet50(pretrained=True).to(self.device)
            module = list(resnet.children())[:6] # best layer -> layer 3
            self.model = torch.nn.Sequential(*module)
            
    def crop_patch(self, img):
        roi = self.roi
        crop_ch = [crop_patch(ch, roi) for ch in img]
        return np.array(crop_ch)
    
    
    def normalize_imgnet(self, img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        mean = np.reshape(mean, (3,1,1))
        std = np.reshape(std, (3,1,1))
        img /= 255
        return (img - mean) / std
    
    
    def pre_process(self, img, inp_shape = (224,224)):
        """
        Crop, normalize and edges smoother (coisine window)
        """
        if self.features_extractor == "alexnet":
            inp_shape = (227,227)
        img_n = self.normalize_imgnet(self.crop_patch(img))
        img_proc = cv2.resize(np.transpose(img_n, (1,2,0)), inp_shape)
        return np.transpose(img_proc, (2,0,1))
    
    
    def pos_process(self, feature_maps):
        def smooth_edge_gpu(img):
            h_h = torch.hann_window(img.shape[-2]) 
            h_w = torch.hann_window(img.shape[-1])
            mask = torch.sqrt(torch.outer(h_h, h_w))
            output = mask.to(self.device) * img 
            return output
        return torch.stack([smooth_edge_gpu(fm) for fm in feature_maps])
    
    
    def start(self, img, bbox, roi):
        assert img.ndim == 3, "This model is for RGB images"
        self.bbox = bbox
        self.roi = roi
        p = self.pre_process(img)
        if self.features_extractor in ["resnet", "mobilenet", "vgg16", "alexnet"]:
            inp = torch.from_numpy(p).unsqueeze(dim = 0).float().to(self.device)
            features = self.model(inp)
            feature_maps = features.squeeze().detach()
            feature_maps_hann = self.pos_process(feature_maps)

        self.X = torch.fft.fftn(feature_maps_hann) 
        y = torch.from_numpy(get_2d_gauss(self.X.shape, sig = self.sig)).to(self.device)
        self.Y = torch.fft.fftn(y)
        self.A = torch.conj(self.Y) * self.X
        self.B = torch.sum(torch.conj(self.X) * self.X, dim = 0)
        

        
    
    def detect(self, img):
        p = self.pre_process(img)
        if self.features_extractor in ["resnet", "mobilenet", "vgg16", "alexnet"]:
            inp = torch.from_numpy(p).unsqueeze(dim = 0).float().to(self.device)
            features = self.model(inp)
            feature_maps = features.squeeze().detach()
            feature_maps_hann = self.pos_process(feature_maps)
            del inp
            del feature_maps
            
        self.X = torch.fft.fftn(feature_maps_hann) 
        
        
        F = self.A / self.B + self.lambda_
        Y = self.X * torch.conj(F)
        self.g = torch.fft.ifftn(torch.sum(Y, dim=0))
        g_cpu = self.g.detach().cpu().numpy()
        loc = np.unravel_index(np.argmax(g_cpu), g_cpu.shape)
        rows = int(loc[0] * self.roi.height/ self.X.shape[-2])
        cols = int(loc[1] * self.roi.width/ self.X.shape[-1])
        
        
        self.bbox, self.roi = transf2ori((rows,cols), self.bbox, self.roi, img.shape[1:])#transform to the ori frame
        
    def update(self):
        self.A = self.lr * torch.conj(self.Y) * self.X + (1 - self.lr) * self.A
        self.B = self.lr * torch.sum(torch.conj(self.X) * self.X, dim = 0) + (1 - self.lr) * self.B

        
        
        
    
class MOSSETracker:
    """ Mosse Tracker variable names according to the slides:
        inputs:
            - lambda: regularization parameter,
            - learning_rate: hyperparameter
            - sigma: for 2D gaussian construction
    """
    def __init__(self, lambda_ = 1e-5, learning_rate = 0.05, sigma = 2):
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.sig = sigma
        self.bbox = None
        self.roi = None
        self.P = 0
        self.A = 0
        self.B = 0
        self.C = 0
        self.g = 0
        
    def crop_patch(self, img):
        return crop_patch(img, self.roi)
        
    def pre_process(self, img):
        """
        Crop, normalize and edges smoother (coisine window)
        """
        img_n = normalize(self.crop_patch(img))
        return smooth_edge(img_n)
    
        
    def start(self, img, bbox, roi):
        assert img.ndim == 2, "This MOSSE is for grayscale images"
        self.bbox = bbox # initial bounding box
        self.roi = roi # searching roi
        p = self.pre_process(img) # paper preprocessing
        
        self.P = fft2(p) # initial patch in the fourier domain

        c = get_2d_gauss(self.P.shape, self.sig)
        self.C = fft2(c) # function target
        
        self.A = np.conj(self.C) * self.P # first frame A (closed form numerator)
        self.B = np.conj(self.P) * self.P # first frame B (closed form denominator)


    def detect(self, img):
        p = self.pre_process(img) # paper pre processing
        self.P = fft2(p) # patch in the fourier domain

        M = self.A / (self.lambda_ + self.B) # filter in the fourier domain
        G = self.P * np.conj(M) # output in the fourier domain
        self.g = ifft2(G).real # output
        loc = np.unravel_index(np.argmax(self.g), self.g.shape) # location of the maximum in the roi (row, col)
        self.bbox, self.roi = transf2ori(loc, self.bbox, self.roi, img.shape) # bbox and roi in the ori cs
        
        
    def update(self):
        self.A = self.lr * np.conj(self.C) * self.P + (1 - self.lr) * self.A # following the slides
        self.B = self.lr * np.conj(self.P) * self.P + (1 - self.lr) * self.B