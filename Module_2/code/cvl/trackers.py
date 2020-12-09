import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
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
        - features: handcrafted/deep
        - lambda: regularization parameter,
        - learning_rate: hyperparameter
        - sigma: for 2D gaussian construction
    """
    def __init__(self, features = "deep_f", lambda_ = 1e-5, learning_rate = 0.125, sigma = 2):
        self.feat_type = features
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.sig = sigma
        
    def crop_patch(self, img):
        roi = self.roi
        return crop_patch(img, roi)
        
    def pre_process(self, img):
        """
        Crop, normalize and edges smoother (coisine window)
        """
        pp_channels = []
        for c in img:
            pp_channels.append(smooth_edge(normalize(self.crop_patch(c))))
        return pp_channels
    
    def start(self, img, bbox, roi):
        assert img.ndim == 3, "This model is for RGB images"
        self.bbox = bbox # initial bounding box
        self.roi = roi # searching roi
        pp_channels = self.pre_process(img) # paper preprocessing
        # TODO GET FEATURE MAPS from deep learning
        self.X = fft2(np.array(pp_channels))
        
        y = get_2d_gauss(self.X.shape[1:], sig = self.sig)
        
        self.Y = fft2(y) # function target
        
        self.A = np.conj(self.Y) * self.X # first frame A (closed form numerator) --> shape (n_channels, h, w)
        self.B = np.sum(np.conj(self.X) * self.X, axis = 0) # first frame B (closed form denominator) 
        
        
    
    def detect(self, img):
        p_channels = self.pre_process(img)
        self.X = fft2(np.array(p_channels))
        
        F = self.A / self.B + self.lambda_
        Y = self.X * np.conj(F)
        self.g = ifft2(np.sum(Y, axis=0)).real
        
        loc = np.unravel_index(np.argmax(self.g), self.g.shape)
        self.bbox, self.roi = transf2ori(loc, self.bbox, self.roi, img.shape[1:])
        
    def update(self):
        self.A = self.lr * np.conj(self.Y) * self.X + (1 - self.lr) * self.A
        self.B = self.lr * np.sum(np.conj(self.X) * self.X, axis = 0) + (1 - self.lr) * self.B
        
        


        