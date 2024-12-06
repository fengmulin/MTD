import functools
import logging
import bisect
from tkinter import E
import threading
import torch.utils.data as data
import torch
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import os
import json
import queue
import scipy.io as scio
from tqdm import tqdm
from PIL import Image

class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])
    processes2 = State(default=[])
    limit = State(default=None)
    load_memory = State(default=False)

    def __init__(self, data_dir=None, data_list=None, limit=None, load_memory=False, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        self.limit = limit 
        self.load_memory = load_memory
        #print(self.limit, "limit********************************************************")
        #raise
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        #q = 
        # self.q = queue.Queue()
        
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.pos_paths = []
        self.get_all_samples()
        if self.load_memory:
            self.img_list = list()
            for item in tqdm(self.image_paths,total=len(self.image_paths)):
                self.img_list.append(self.load_img(item))


    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            # print(1111)
            # raise
            with open(self.data_list[i], 'r') as fid:
                if not self.limit:
                    image_list = fid.readlines()
                    #print(type(image_list), len(image_list))
                else:
                    image_list = fid.readlines()
                    lens = len(image_list)
                    #print(not self.limit, self.limit, 44444444444444444444444444)
                    temp = int(self.limit*lens)
                    image_list = image_list[:temp]
                    
            if self.is_training:
                if "2017" in self.data_dir[i]:
                    image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                    gt_path=[self.data_dir[i]+'/train_gts/'+'gt_'+timg.strip()[:-4]+'.txt' for timg in image_list]
                    #pos_path=[self.data_dir[i]+'/train_pos/'+timg.strip()[:-4]+'.txt' for timg in image_list]
                elif "CTW" in self.data_dir[i] or 'tpa' in self.data_dir[i] or 'syn_curve' in self.data_dir[i] or 'ASAYAR' in self.data_dir[i]:
                    image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                    gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()[:-4]+'.txt' for timg in image_list]
                elif 'syn_MPSC' in self.data_dir[i]:
                    for ii in range(0,1132):
                        temp_dir = self.data_dir[i] +'gen_pictures/'+ str(ii) + '/'
                        qzz = os.listdir(temp_dir)
                        for iii in qzz:

                            self.image_paths.append(temp_dir+iii)
                            self.gt_paths.append(self.data_dir[i] + 'gen_labels/' + str(ii) + '/'+ iii[:-4] +'.txt')
                elif "MPSC" in self.data_dir[0]:
                    image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                    gt_path=[self.data_dir[i]+'/train_gts/gt_'+timg.strip()[5:-4]+'.txt' for timg in image_list]
                elif 'Synth' in self.data_dir[i]:
                    data = scio.loadmat(self.data_dir[i]+'/gt.mat')

                    self.image_paths = data['imnames'][0]
                    self.gts = data['wordBB'][0]
                    self.texts = data['txt'][0]
                elif 'synth_curve' in self.data_dir[0]:
                    # print(self.data_dir)
                    # raise
                    with open(self.data_dir[0] +'merge_v2.json', 'r') as file:
                        # 读取 JSON 数据
                        data = json.load(file)
                    self.image_paths = data['img_path']
                    # print(self.image_paths[0])
                    # raise
                    self.gts = data['polygon']
                    # self.texts = data['txt'][0]
                else:

                    image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                    gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()+'.txt' for timg in image_list]
            else:
                #continue
                # print('yes',self.data_list[i])
                # raise
                image_path=[self.data_dir[i]+'/test_images/'+timg.strip() for timg in image_list]
                # print(self.data_list[i])
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    # print(33)
                    # raise
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()+'.txt' for timg in image_list]
                elif 'CTW1500' in self.data_list[i] or 'tpa' in self.data_list[i] or 'ASAYAR'  in self.data_dir[i]:
                    # print(44,self.data_list[i])
                    # raise
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip().split('.')[0]+'.txt' for timg in image_list]
                elif 'TGPT'  in self.data_list[i]:
                    # print(11)
                    # raise
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip().split('.')[0]+'.jpg.txt' for timg in image_list]
                elif 'MPSC' in self.data_list[i]:

                    gt_path=[self.data_dir[i]+'/test_gts/gt_'+timg.strip().split('.')[0][5:]+'.txt' for timg in image_list]
                elif "2017" in self.data_dir[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt'+timg.strip().split('.')[0][2:]+'.txt' for timg in image_list]
                else:
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt_'+timg.strip().split('.')[0]+'.txt' for timg in image_list]
            if not 'Synth' in self.data_dir[i] and not 'syn_MPSC' in self.data_dir[i] and not 'synth_curve' in self.data_dir[i]:
                 
                self.image_paths += image_path
                self.gt_paths += gt_path
 
        if not 'Synth' in self.data_dir[0] and not 'synth_curve' in self.data_dir[0]:
            self.num_samples = len(self.image_paths)
            self.targets = self.load_ann()
            #self.images = self.img_read.run()
            if self.is_training:
                # print(len(self.image_paths), len(self.targets))
                assert len(self.image_paths) == len(self.targets)
        else:
            self.num_samples = len(self.image_paths)
    def load_img(self, image_path):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
            # print(img.shape,11)
        except:
            print(image_path,99999)
            img = np.array(Image.open(image_path))
            img = img[:, :, [2, 1, 0]]
        return img
    
    def load_ann(self):
        res = []
        lens =len(self.gt_paths)
        for  gt in tqdm(self.gt_paths, total=lens):
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:

                item = {}
                parts = line.strip().split(',')
                label = parts[-1]

                if 'TD' in self.data_dir[0] and label == '1':
                    label = '###'
                if 'icdar' in self.data_dir[0]:
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif '2017' in gt:
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif 'MPSC' in self.data_dir[0]:
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif 'CTW' in self.data_dir[0]:
                    label  = label.replace('"','')
                    if not self.is_training:
                        if '#' in label:
                            label = '111'
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    poly = np.array(list(map(float, line[:28]))).reshape((-1, 2)).tolist()
                elif 'Syn' in self.data_dir[0]:
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                else:
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                # print(poly, len(poly))
                if len(poly)<3:
                    continue
                item['poly'] = np.array(poly)
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res
    def get_ann_synth(self,img,index,image_path):
        bboxes = np.array(self.gts[index])
        bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
        bboxes = bboxes.transpose(2, 1, 0)
        words = []
        for text in self.texts[index]:
            text = text.replace('\n', ' ').replace('\r', ' ')
            words.extend([w for w in text.split(' ') if len(w) > 0])
        lines = []
        for box,word in zip(bboxes,words):
            z={'poly':(box+0.5).tolist(), 'text':word}
            lines.append(z)
        return lines
    def get_ann_syn_curve(self,img,index,image_path):
        # print(self.gts[index])
        # raise
        bboxes = self.gts[index]
        lines = []
        for box in zip(bboxes):
            # print(box)
            # try:
                # print(index)
                # print(box,1111)
            gtt = str(box[0]).split(',')[:-1]
            gtt = np.array(list(map(float, gtt))).reshape((-1, 2))
            if len(gtt)<4:
                # prin
                continue
            z={'poly':gtt, 'text':'qzz'}
            lines.append(z)
        return lines
    def __getitem__(self, index, retry=0):
        # if index >= self.num_samples:
        #     index = index % self.num_samples
        data = {}
        if self.load_memory:
            data['image'] = self.img_list[index]
            image_path = self.image_paths[index]
        else:
            image_path = self.image_paths[index]
            # print(1111,self.data_dir[0])
            if 'synth_curve' in self.data_dir[0]:
                # try:
                image_path = os.path.join(self.data_dir[0],'train_images/'+image_path )
                # image_path = os.path.join(self.data_dir[0],'train_images/'+image_path )
                if '\n' in image_path:
                    image_path = image_path[:-1]
            if 'Synth' in self.data_dir[0]:

                image_path = os.path.join(self.data_dir[0],image_path[0] )      
            try:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
            except:
                img = np.array(Image.open(image_path))
                img = img[:, :, [2, 1, 0]]

            data['image'] = img
        data['is_training'] = True if 'train' in self.data_list[0] else False
        # print(data['is_training'])
        if not 'Synth' in self.data_dir[0] and not 'synth_curve' in self.data_dir[0]:
            if self.is_training:
                data['filename'] = image_path
                data['data_id'] = image_path
            else:
                data['filename'] = image_path.split('/')[-1]
                data['data_id'] = image_path.split('/')[-1]
            target = self.targets[index]
            data['lines'] = target
        elif  'synth_curve' in self.data_dir[0]:
            if self.is_training:
                data['filename'] = image_path
                data['data_id'] = image_path
            else:
                data['filename'] = image_path.split('/')[-1]
                data['data_id'] = image_path.split('/')[-1]
            data['lines'] = self.get_ann_syn_curve(img,index,image_path)
        else:
            if self.is_training:
                data['filename'] = image_path
                data['data_id'] = image_path
            else:
                data['filename'] = image_path.split('/')[-1]
                data['data_id'] = image_path.split('/')[-1]
            data['lines'] = self.get_ann_synth(img,index,image_path)
        data['shape'] = data['image'].shape[:2]
        data['polygon'] = [p for p in data['lines']]

        # print(data['polygon'])
        # raise

        if self.processes is not None:
            for data_process in self.processes:
                data['image'], data['polygon'] = data_process(data['image'], data['polygon'])

        if self.processes2 is not None:
            for data_process in self.processes2:
                data = data_process(data)

        data['image'] = data['image'].transpose(2,0,1)
     
        return data

    def __len__(self):
        if not 'Synth' in self.data_dir[0]:
            return len(self.image_paths)
        else:
            return self.image_paths.shape[0]

