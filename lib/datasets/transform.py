# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import cv2
import numpy as np
import torch
from scipy import ndimage
from PIL import Image, ImageEnhance

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_continuous=False,fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h,w):
            return sample
            
        if self.fix:
            h_rate = self.output_size[0]/h
            w_rate = self.output_size[1]/w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else: 
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)
        
        top = (self.output_size[0] - new_h)//2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w)//2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img = cv2.copyMakeBorder(img,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0,0,0])  

        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation'] 
            seg = cv2.resize(segmentation, dsize=(new_w,new_h), interpolation=self.seg_interpolation)
            if self.fix:
                seg = cv2.copyMakeBorder(seg,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg

        if 'edge' in sample.keys():
            edge = sample['edge'] 
            edge = cv2.resize(edge, dsize=(new_w,new_h), interpolation=self.seg_interpolation)
            if self.fix:
                edge = cv2.copyMakeBorder(edge,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0])
            sample['edge'] = edge

        sample['image'] = img
        return sample

class Centerlize(object):
    def __init__(self, output_size, is_continuous=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h,w):
            return sample

        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        
        top = (new_h - h) // 2  
        bottom = new_h - h - top
        left = (new_w - w) // 2
        right = new_w - w -left
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])   
        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation'] 
            seg=cv2.copyMakeBorder(segmentation,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
            sample['segmentation'] = seg
        if 'edge' in sample.keys():
            edge = sample['edge'] 
            edge=cv2.copyMakeBorder(edge,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
            sample['edge'] = edge
        sample['image'] = img
        
        return sample
                     
class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint((h-new_h)//4, (h-new_h)*3//4+1)
        left = np.random.randint((w-new_w)//4, (w-new_w)*3//4+1)
    #    top = np.random.randint(0, h - new_h + 1)
    #    left = np.random.randint(0, w - new_w + 1)
    #    top = (h - new_h) // 2  
    #    left = (w - new_w) // 2  

        image = image[top: top + new_h,
                      left: left + new_w]

        segmentation = segmentation[top: top + new_h,
                      left: left + new_w]
        if 'edge' in sample.keys():
            edge = sample['edge']
            edge = edge[top: top + new_h,
                      left: left + new_w]
            sample['edge'] = edge

        sample['image'] = image
        sample['segmentation'] = segmentation
        return sample
                     
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        segmentation = segmentation[top: top + new_h,
                      left: left + new_w]
        if 'edge' in sample.keys():
            edge = sample['edge']
            edge = edge[top: top + new_h,
                      left: left + new_w]
            sample['edge'] = edge

        sample['image'] = image
        sample['segmentation'] = segmentation
        return sample

class RandomHSV(object):
    """Generate randomly the image in hsv space."""
    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        image = sample['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:,:,0].astype(np.int32)
        s = hsv[:,:,1].astype(np.int32)
        v = hsv[:,:,2].astype(np.int32)
        delta_h = np.random.randint(-self.h_r,self.h_r)
        delta_s = np.random.randint(-self.s_r,self.s_r)
        delta_v = np.random.randint(-self.v_r,self.v_r)
        h = (h + delta_h)%180
        s = s + delta_s
        s[s>255] = 255
        s[s<0] = 0
        v = v + delta_v
        v[v>255] = 255
        v[v<0] = 0
        hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)	
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        sample['image'] = image
        return sample

class RandomFlip(object):
    """Randomly flip image"""
    def __init__(self, threshold):
        self.flip_t = threshold
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        if np.random.rand() < self.flip_t:
            image_flip = np.flip(image, axis=1)
            segmentation_flip = np.flip(segmentation, axis=1)
            sample['image'] = image_flip
            sample['segmentation'] = segmentation_flip

        if 'edge' in sample.keys():
            edge = sample['edge']
            edge = np.flip(edge, axis=1)
            sample['edge'] = edge

        return sample

class RandomRotation(object):
    """Randomly rotate image"""
    def __init__(self, angle_r, is_continuous=False):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
        new_image = cv2.warpAffine(image, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=0)
        new_segmentation = cv2.warpAffine(segmentation, m, (col,row), flags=self.seg_interpolation, borderValue=0)
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation

        if 'edge' in sample.keys():
            edge = sample['edge']
            edge = cv2.warpAffine(edge, m, (col,row), flags=self.seg_interpolation, borderValue=0)
            sample['edge'] = edge

        return sample

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix
    
class RandomShearX(object):
    """Randomly shear image along x"""
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, sample):
        magnitudes = np.linspace(-0.3, 0.3, self.magnitude)
        rand_magnitude = np.random.randint(0, self.magnitude-1)
        image, segmentation = sample['image'], sample['segmentation']
        transform_matrix = np.array([[1, np.random.uniform(magnitudes[rand_magnitude], magnitudes[rand_magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
        transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
        affine_matrix = transform_matrix[:2, :2]
        offset = transform_matrix[:2, 2]
        new_image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
        new_segmentation = ndimage.interpolation.affine_transform(segmentation, affine_matrix, offset, order=0, mode='nearest')
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation

        if 'edge' in sample.keys():
            edge = sample['edge']
            new_edge = ndimage.interpolation.affine_transform(edge, affine_matrix, offset, order=0, mode='nearest')
            sample['edge'] = new_edge

        return sample

class RandomShearY(object):
    """Randomly shear image along x"""
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, sample):
        magnitudes = np.linspace(-0.3, 0.3, self.magnitude)
        rand_magnitude = np.random.randint(0, self.magnitude-1)
        image, segmentation = sample['image'], sample['segmentation']
        transform_matrix = np.array([[1, 0, 0],
                                 [np.random.uniform(magnitudes[rand_magnitude], magnitudes[rand_magnitude+1]), 1, 0],
                                 [0, 0, 1]])
        transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
        affine_matrix = transform_matrix[:2, :2]
        offset = transform_matrix[:2, 2]
        new_image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
        new_segmentation = ndimage.interpolation.affine_transform(segmentation, affine_matrix, offset, order=0, mode='nearest')
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation

        if 'edge' in sample.keys():
            edge = sample['edge']
            new_edge = ndimage.interpolation.affine_transform(edge, affine_matrix, offset, order=0, mode='nearest')
            sample['edge'] = new_edge

        return sample

class RandomScale(object):
    """Randomly scale image"""
    def __init__(self, scale_r, is_continuous=False):
        self.scale_r = scale_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_scale = np.random.rand()*(self.scale_r - 1/self.scale_r) + 1/self.scale_r
    #    rand_scale = np.random.rand()*(self.scale_r - 1/self.scale_r) + 2/self.scale_r - 1.
        img = cv2.resize(image, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(segmentation, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['segmentation'] = seg

        if 'edge' in sample.keys():
            edge = sample['edge']
            edge = cv2.resize(edge, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
            sample['edge'] = edge

        return sample

class Multiscale(object):
    def __init__(self, rate_list):
        self.rate_list = rate_list

    def __call__(self, sample):
        image = sample['image']
        row, col, _ = image.shape
        image_multiscale = []
        for rate in self.rate_list:
            rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            sample['image_%f'%rate] = rescaled_image
        return sample

class RandomContrast(object):
    def __init__(self, magnitudes=np.linspace(0.5, 1., 11)):
        self.magnitudes = magnitudes

    def __call__(self, sample):
        rand_magnitude = np.random.randint(0, 11-1)
        image = sample['image']
        image = Image.fromarray(image)
        image_new = ImageEnhance.Contrast(image).enhance(np.random.uniform(self.magnitudes[rand_magnitude], self.magnitudes[rand_magnitude+1]))
        sample['image'] = np.asarray(image_new)
        return sample

class RandomBrightness(object):
    def __init__(self, magnitudes=np.linspace(0.5, 1., 11)):
        self.magnitudes = magnitudes

    def __call__(self, sample):
        rand_magnitude = np.random.randint(0, 11-1)
        image = sample['image']
        image = Image.fromarray(image)
        image_new = ImageEnhance.Brightness(image).enhance(np.random.uniform(self.magnitudes[rand_magnitude], self.magnitudes[rand_magnitude+1]))
        sample['image'] = np.asarray(image_new)
        return sample

class RandomSharpness(object):
    def __init__(self, magnitudes=np.linspace(0.1, 3., 11)):
        self.magnitudes = magnitudes

    def __call__(self, sample):
        rand_magnitude = np.random.randint(0, 11-1)
        image = sample['image']
        image = Image.fromarray(image)
        image_new = ImageEnhance.Sharpness(image).enhance(np.random.uniform(self.magnitudes[rand_magnitude], self.magnitudes[rand_magnitude+1]))
        sample['image'] = np.asarray(image_new)
        return sample

class RandomColor(object):
    def __init__(self, magnitudes=np.linspace(0.5, 1., 11)):
        self.magnitudes = magnitudes

    def __call__(self, sample):
        rand_magnitude = np.random.randint(0, 11-1)
        image = sample['image']
        image = Image.fromarray(image)
        image_new = ImageEnhance.Color(image).enhance(np.random.uniform(self.magnitudes[rand_magnitude], self.magnitudes[rand_magnitude+1]))
        sample['image'] = np.asarray(image_new)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key:
                image = sample[key]
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image = image.transpose((2,0,1))
                sample[key] = torch.from_numpy(image.astype(np.float32)/255.0)
                #sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
            elif 'segmentation' == key:
                segmentation = sample['segmentation']
                sample['segmentation'] = torch.from_numpy(segmentation.astype(np.float32))
            elif 'edge' == key:
                edge = sample['edge']
                sample['edge'] = torch.from_numpy(edge.astype(np.float32))
            elif 'segmentation_onehot' == key:
                onehot = sample['segmentation_onehot'].transpose((2,0,1))
                sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.float32))
            elif 'mask' == key:
                mask = sample['mask']
                sample['mask'] = torch.from_numpy(mask.astype(np.float32))
        return sample

def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot
