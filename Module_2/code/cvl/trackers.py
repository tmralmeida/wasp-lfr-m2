import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
from .lib import normalize, smooth_edge, transf2ori


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
    def __init__(self, lambda_ = 1e-5, learning_rate = 0.125, sigma = 2):
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
        # Centered 2D gaussian based on https://www.geeksforgeeks.org/how-to-generate-2-d-gaussian-array-using-numpy/
        x, y = np.meshgrid(np.linspace(-10, 10, self.P.shape[1]),
                           np.linspace(-10, 10, self.P.shape[0]))
        dst = np.sqrt (x**2 + y**2)
        c = np.exp(-((dst**2) / (2.0 * self.sig**2 )))
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
        

        
        
        


        