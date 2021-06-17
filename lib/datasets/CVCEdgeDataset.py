# ----------------------------------------
# Written by Xiaoqing GUO
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *

def generate_noise_data(label, w, h, noise_percent):
    Prob_map = np.arange(0.,w*h,1.)
    np.random.shuffle(Prob_map)
    Prob_map = Prob_map.reshape((w, h))
    Prob_map = np.where(Prob_map < w*h*noise_percent, 1, 0)
    Prob_map = Prob_map > 0
    noise_label = np.where(Prob_map, 1.-label, label)
    return noise_label

class CVCEdgeDataset(Dataset):
    def __init__(self, dataset_name, cfg, period, aug=False):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data')
        self.dataset_dir = os.path.join(self.root_dir,dataset_name)
        self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.period = period
        self.img_dir = os.path.join(self.dataset_dir, 'images')
        self.edge_dir = os.path.join(self.dataset_dir, 'boundary')
        self.seg_dir = os.path.join(self.dataset_dir, 'labels')
        self.set_dir = os.path.join(self.root_dir, dataset_name)
        file_name = None
    #    if aug:
    #        file_name = self.set_dir+'/'+period+'aug.txt'
    #    else:
    #        file_name = self.set_dir+'/'+period+'.txt'
    #    df = pd.read_csv(file_name, names=['filename'])
    #    self.name_list = df['filename'].values
        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomshearx = None
        self.randomsheary = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.totensor = ToTensor()
        self.update_list()
        self.cfg = cfg
	
        if dataset_name == 'CVC':
            self.categories = ['Polyp'] 
            self.coco2voc = [[0],
                             [5],
                             [2],
                             [16],
                             [9],
                             [44],#,46,86],
                             [6],
                             [3],#,8],
                             [17],
                             [62],
                             [21],
                             [67],
                             [18],
                             [19],#,24],
                             [4],
                             [1],
                             [64],
                             [20],
                             [63],
                             [7],
                             [72]]

            self.num_categories = len(self.categories)
            assert(self.num_categories+1 == self.cfg.MODEL_NUM_CLASSES)
            self.cmap = self.__colormap(len(self.categories)+1)


        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE,fix=False)
            #self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:        
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSHEAR > 0:
                self.randomshearx = RandomShearX(cfg.DATA_RANDOMSHEAR)
                self.randomsheary = RandomShearY(cfg.DATA_RANDOMSHEAR)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx].split()[0]
        img_file = self.img_dir + '/' + name 
    #    if self.period == 'aug_train':
    #        print(img_file)
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}
        
        if 'train' in self.period:
            seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]
            edge_file = self.edge_dir + '/' + self.name_list[idx].split()[0]
            segmentation = np.array(Image.open(seg_file))
            (T, segmentation) = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
            sample['segmentation'] = segmentation/255.

            edge = np.array(Image.open(edge_file))
            (T, edge) = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY)
            sample['edge'] = edge/255.

        #    sample['segmentation'] = generate_noise_data(segmentation/255., r, c, self.cfg.noise_percent)
        #    if self.period == 'train':
        #        infile = open('/home/xiaoqiguo2/AASPL/log.txt', 'a')
        #        infile.write(str(sample['segmentation'].shape))
            
            if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSHEAR > 0:
                sample = self.randomshearx(sample)
                sample = self.randomsheary(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                #sample = self.centerlize(sample)
                sample = self.rescale(sample)
        else:
            seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]
            segmentation = np.array(Image.open(seg_file))
            (T, segmentation) = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
            sample['segmentation'] = segmentation/255.

            edge_file = self.edge_dir + '/' + self.name_list[idx].split()[0]
            edge = np.array(Image.open(edge_file))
            (T, edge) = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY)
            sample['edge'] = edge/255.

            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        if 'segmentation' in sample.keys():
            sample['mask'] = sample['segmentation'] < self.cfg.MODEL_NUM_CLASSES
            #print(sample['segmentation'].max(),sample['segmentation'].shape)
            t = sample['segmentation']
            t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            #print(t.max(),t.shape)
            #print(onehot(np.int32(t),self.cfg.MODEL_NUM_CLASSES))
            sample['segmentation_onehot']=onehot(np.int32(t),self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample

    def update_list(self):
        file_name = self.set_dir+'/'+self.period+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        name_list = df['filename'].values 
        self.name_list = np.unique(name_list)
    #    print('list has been updated!') 
    #    print(self.name_list)

    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap
    
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        return cmap
    
    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s'%sample['name'])
            # predict_color = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(file_path, sample['predict'])
            # print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def save_result_train(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_test_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s'%sample['name'])

            img_file = self.img_dir + '/' + sample['name'] 
            input_img = cv2.imread(img_file)
            input_img = cv2.resize(input_img, (sample['predict'].shape[0],sample['predict'].shape[1]))
            pred_img = np.stack([sample['predict'], sample['predict'], sample['predict']]).transpose((1,2,0))
            lab_img = np.stack([sample['label'], sample['label'], sample['label']]).transpose((1,2,0))

            img=np.zeros([input_img.shape[0], input_img.shape[1]*3, input_img.shape[2]])
            img[:,:input_img.shape[1], :] = input_img
            img[:,input_img.shape[1]:input_img.shape[1]*2, :] = pred_img
            img[:,input_img.shape[1]*2:, :] = lab_img
            cv2.imwrite(file_path, img)
            # print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def save_result_train_weight(self, result_list, model_id):
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s'%sample['name'])

            img_file = self.img_dir + '/' + sample['name'] 
            #input_img = cv2.imread(img_file)
            input_img = sample['input'].transpose((1,2,0))
            pred_img = np.stack([sample['predict'], sample['predict'], sample['predict']]).transpose((1,2,0))
            lab_img = np.stack([sample['label'], sample['label'], sample['label']]).transpose((1,2,0))
            v_class1 = np.stack([sample['v_class1'], sample['v_class1'], sample['v_class1']]).transpose((1,2,0))
            v_class0 = np.stack([sample['v_class0'], sample['v_class0'], sample['v_class0']]).transpose((1,2,0))

            img=np.zeros([input_img.shape[0], input_img.shape[1]*5, input_img.shape[2]])
            img[:,:input_img.shape[1], 0] = input_img[:,:,2]
            img[:,:input_img.shape[1], 1] = input_img[:,:,1]
            img[:,:input_img.shape[1], 2] = input_img[:,:,0]
            img[:,input_img.shape[1]:input_img.shape[1]*2, :] = pred_img
            img[:,input_img.shape[1]*2:input_img.shape[1]*3, :] = lab_img
            img[:,input_img.shape[1]*3:input_img.shape[1]*4, :] = v_class1
            img[:,input_img.shape[1]*4:input_img.shape[1]*5, :] = v_class0
            cv2.imwrite(file_path, img)
            # print('[%d/%d] %s saved'%(i,len(result_list),file_path))

    def do_matlab_eval(self, model_id):
        import subprocess
        path = os.path.join(self.root_dir, 'VOCcode')
        eval_filename = os.path.join(self.eval_dir,'%s_result.mat'%model_id)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; VOCinit; '
        cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
        cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
        cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
        cmd += 'quit;"'

        print('start subprocess for matlab evaluation...')
        print(cmd)
        subprocess.call(cmd, shell=True)
    
    def do_python_eval(self, model_id):
        predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        gt_folder = self.seg_dir
        TP = []
        P = []
        T = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))
        
        '''def compare(start,step,TP,P,T):
            for idx in range(start,len(self.name_list),step):
                print('%d/%d'%(idx,len(self.name_list)))
                name_image = self.name_list[idx].split()[0]
                name_seg = self.name_list[idx].split()[1]
                predict_file = os.path.join(predict_folder,'%s'%name_image)
                gt_file = os.path.join(gt_folder,'%s'%name_seg)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                gt = np.array(Image.open(gt_file))
                (_, predict) = cv2.threshold(predict, 0, 255, cv2.THRESH_BINARY)
                #print(np.unique(predict), np.unique(gt))
                predict = np.int32(predict/255.)
                gt = np.int32(gt/255.)
                cal = gt<255
                mask = (predict==gt) * cal
          
                for i in range(self.cfg.MODEL_NUM_CLASSES):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask)
                    TP[i].release()
        p_list = []
        for i in range(15):
            p = multiprocessing.Process(target=compare, args=(i,15,TP,P,T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            print(TP[i].value, T[i].value, P[i].value)
            IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%11s:%7.3f%%'%('backbound',IoU[i]*100),end='\t')
            else:
                if i%2 != 1:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100),end='\t')
                else:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
                    
        miou = np.mean(np.array(IoU))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100)) '''   

    def do_python_eval(self, model_id):
        IoU_array = 0.
        sample_num = 0.
        predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        gt_folder = self.seg_dir
        TP = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        P = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        T = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        for idx in range(len(self.name_list)):
         #   print('%d/%d'%(idx,len(self.name_list)))
            name_image = self.name_list[idx].split()[0]
            name_seg = self.name_list[idx].split()[0]
            predict_file = os.path.join(predict_folder,'%s'%name_image)
            gt_file = os.path.join(gt_folder,'%s'%name_seg)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            gt = np.array(Image.open(gt_file))
            (_, predict) = cv2.threshold(predict, 0, 255, cv2.THRESH_BINARY)
            predict = np.int32(predict/255.)
            gt = np.int32(gt/255.)
            cal = gt<255
            mask = (predict==gt) * cal
            #print(np.unique(predict), np.unique(gt), np.unique(cal), np.unique(mask))
                
            for i in range(self.cfg.MODEL_NUM_CLASSES):
                P[i] = np.sum((predict==i))
                T[i] = np.sum((gt==i))
                TP[i] = np.sum((gt==i)*mask)
            TP = TP.astype(np.float64)
            T = T.astype(np.float64)
            P = P.astype(np.float64)
            IoU = TP/(T+P-TP)
            IoU_array += IoU
            sample_num += 1
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%15s:%7.3f%%'%('background',IoU_array[i]*100/sample_num))
            else:
                print('%15s:%7.3f%%'%(self.categories[i-1],IoU_array[i]*100/sample_num))
        miou = np.mean(IoU_array/sample_num)
        print('==================================')
        print('%15s:%7.3f%%'%('mIoU',miou*100))

    def __coco2voc(self, m):
        r,c = m.shape
        result = np.zeros((r,c),dtype=np.uint8)
        for i in range(0,21):
            for j in self.coco2voc[i]:
                result[m==j] = i
        return result
