import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import torch
import torchvision.models as models
import cv2
from .image_io import crop_patch
from .lib import normalize, smooth_edge, get_2d_gauss, transf2ori
from .features import alexnetFeatures


class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        responsef = np.conj(self.template) * patchf
        response = ifft2(responsef)

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr

class MOSSETracker:
    """ Mosse Tracker variable names according to the slides:
        inputs:
            - lambda: regularization parameter,
            - learning_rate: hyperparameter
            - sigma: for 2D gaussian construction
    """
    def __init__(self, features = "deep", lambda_ = 1e-5, learning_rate = 0.125, sigma = 2):
        self.feat_extractor = features
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
        roi = self.roi
        return crop_patch(img, roi)
        
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

        M = self.A / (self.lambda_ * self.B) # filter in the fourier domain
        G = self.P * np.conj(M) # output in the fourier domain
        self.g = ifft2(G).real # output
        
        loc = np.unravel_index(np.argmax(self.g), self.g.shape) # location of the maximum in the roi (row, col)
        self.bbox, self.roi = transf2ori(loc, self.bbox, self.roi, img.shape) # bbox and roi in the ori cs
        
    def update(self):
        self.A = self.lr * np.conj(self.C) * self.P + (1 - self.lr) * self.A # following the slides
        self.B = self.lr * np.conj(self.P) * self.P + (1 - self.lr) * self.B
        
        
class DCFMOSSETracker:
    """ Mosse Tracker variable names according to the slides:
    inputs:
        - feature_extractor: handcrafted or DL
        - lambda: regularization parameter,
        - learning_rate: hyperparameter
        - sigma: for 2D gaussian construction
    """
    def __init__(self, 
                 features = "alexnet",
                 lambda_ = 1e-5, 
                 learning_rate = 0.125, 
                 sigma = 2):
        
        self.features_extractor = features
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.sig = sigma
        if self.features_extractor == "alexnet":
            self.model = alexnetFeatures(pretrained=True, progress = False)
        elif self.features_extractor == "vgg16":
            self.model = models.vgg16(pretrained=True).features[:2]
        elif self.features_extractor == "mobilenet":
            self.model = models.mobilenet_v2(pretrained=True).features[:2]
        elif self.features_extractor == "resnet":
            resnet = models.resnet50(pretrained=True)
            module = list(resnet.children())[:3]
            self.model = torch.nn.Sequential(*module)
            
    def crop_patch(self, img):
        roi = self.roi
        crop_ch = [crop_patch(ch, roi) for ch in img]
        return np.array(crop_ch)
    
    
    def normalize_imgnet(self, img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        mean = np.reshape(mean, (3,1,1))
        std = np.reshape(std, (3,1,1))
        img /= 255
        return (img - np.array(mean)) / np.array(std)
    
    
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
        return np.array([smooth_edge(fm) for fm in feature_maps])
    
    
    def start(self, img, bbox, roi):
        assert img.ndim == 3, "This model is for RGB images"
        self.bbox = bbox # initial bounding box
        self.roi = roi # searching roi
        p = self.pre_process(img) # paper preprocessing
        if self.features_extractor in ["resnet", "mobilenet", "vgg16", "alexnet"]:
            inp = torch.from_numpy(p).unsqueeze(dim = 0).float()
            features = self.model(inp)
            feature_maps = features.squeeze().detach().numpy()
        feature_maps_h = self.pos_process(feature_maps) # applying cosine window
        self.X = np.array([fft2(fm) for fm in feature_maps_h])
        
        y = get_2d_gauss(self.X.shape, sig = self.sig)
        
        self.Y = fft2(y) # function target
        self.A = np.conj(self.Y) * self.X # first frame A (closed form numerator) --> shape (n_channels, h, w)
        self.B = np.sum(np.conj(self.X) * self.X, axis = 0) # first frame B (closed form denominator) 
        
        
    
    def detect(self, img):
        p = self.pre_process(img) # paper preprocessing
        if self.features_extractor in ["resnet", "mobilenet", "vgg16", "alexnet"]:
            inp = torch.from_numpy(np.array(p)).unsqueeze(dim = 0).float()
            features = self.model(inp)
            feature_maps = features.squeeze().detach().numpy()
        feature_maps_h = self.pos_process(feature_maps) # applying cosine window
        self.X = np.array([fft2(fm) for fm in feature_maps_h])  
              
        F = self.A / self.B + self.lambda_
        Y = self.X * np.conj(F)
        self.g = ifft2(np.sum(Y, axis=0)).real
        loc = np.unravel_index(np.argmax(self.g), self.g.shape) # regarding the feature map (row, col)
        rows = int(loc[0] * self.roi.height/ self.X.shape[-2])
        cols = int(loc[1] * self.roi.width/ self.X.shape[-1])
        
        self.bbox, self.roi = transf2ori((rows,cols), self.bbox, self.roi, img.shape[1:])#transform to the ori frame
        
    def update(self):
        self.A = self.lr * np.conj(self.Y) * self.X + (1 - self.lr) * self.A
        self.B = self.lr * np.sum(np.conj(self.X) * self.X, axis = 0) + (1 - self.lr) * self.B
        
        
        