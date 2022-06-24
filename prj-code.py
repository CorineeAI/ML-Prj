#!/usr/bin/env python
# coding: utf-8

# # 2022 머신러닝 프로젝트(판교제로시티) 

# ## 주제 : Mask Rcnn을 활용한 횡단보도 segmentation

# ### Data : JPG, JSON (COCO FORAMT)
# ### Model : Mask Rcnn (by Dacon)
# * https://dacon.io/competitions/official/235855/codeshare/3725?page=1&dtype=recent

# * cuda, cudnn 호환문제로 tensorflow 사용불가..(컴퓨터 포맷하고 다시 설치할 것)

# In[1]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import base64
import time
import math
import datetime
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from glob import glob


# In[2]:


import torch
import torchvision
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import defaultdict, deque


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


# 데이터 로드


# In[5]:


import natsort #텍스트를 쉽게 정렬할 수 있다.
# 사용방법: natsort.natsorted(seq, key=None, reverse=False, alg=0)

train_files = sorted(glob('./desktop/examples/semantic_segmentation/data_sidewalk/*'))
train_files = natsort.natsorted(train_files)

test_files = sorted(glob('./desktop/examples/semantic_segmentation/test_set/*'))
test_files = natsort.natsorted(test_files)


# In[6]:


train_json_list = []
for file in tqdm(train_files):
    if file.endswith('json'):
        with open(file, "r") as json_file:
            train_json_list.append(json.load(json_file))


# In[7]:


test_json_list = []
for file in tqdm(test_files):
    if file.endswith('json'):
        with open(file, "r") as json_file:
            test_json_list.append(json.load(json_file))


# In[8]:


# EDA, & 시각화


# In[9]:


label_count = {}
for data in train_json_list:
    for shape in data['shapes']:
        try:
            label_count[shape['label']]+=1
        except:
            label_count[shape['label']]=1


# In[10]:


train_jpg = [i for i in train_files if i.endswith('jpg')]
train_json = [i for i in train_files if i.endswith('json')]


# In[11]:


test_jpg = [i for i in test_files if i.endswith('jpg')]
test_json = [i for i in test_files if i.endswith('json')]


# In[12]:


plt.figure(figsize=(30,30))
for i,j in enumerate(range(0,4000,100)):
    plt.subplot(6,5,i+1)
    
    # base64 형식을 array로 변환
    #img = Image.open(BytesIO(base64.b64decode(train_json_list[i]['imageData'])))
    img = Image.open(train_jpg[j])
    img = np.array(img, np.uint8)
    title = []
    for shape in train_json_list[j]['shapes']:
        points = np.array(shape['points'], np.int32)
        cv2.polylines(img, [points], True, (0,255,0), 3)
        title.append(shape['label'])
    title = ','.join(title)
    plt.imshow(img)
    plt.subplot(6,5,i+1).set_title(title)
    if i == 29: break
plt.show()


# In[13]:


plt.figure(figsize=(30,30))
for i,j in enumerate(range(0,4000,100)):
    plt.subplot(6,5,i+1)
    
    # base64 형식을 array로 변환
    #img = Image.open(BytesIO(base64.b64decode(train_json_list[i]['imageData'])))
    img = Image.open(train_jpg[j])
    img = np.array(img, np.uint8)
    title = []
    for shape in train_json_list[j]['shapes']:
        points = np.array(shape['points'], np.int32)
        cv2.polylines(img, [points], True, (0,255,0), 3)
        title.append(shape['label'])
        
    img=Image.fromarray(img)
    Image_sidewalk = img.crop((min(points[:,0]), min(points[:,1]), max(points[:,0]), max(points[:,1])))

    title = ','.join(title)
    plt.imshow(Image_sidewalk)
    plt.subplot(6,5,i+1).set_title(title)
    if i == 29: break
plt.show()


# In[ ]:





# In[ ]:


# 데이콘 데이터에는 이미지 데이터가 base64형식으로 json 파일에 포함되어 있다고 한다.
# Dacon data version
# ---------------------------
# plt.figure(figsize=(25,30))
# for i in range(30):
#     plt.subplot(6,5,i+1)
#     # base64 형식을 array로 변환
#     img = Image.open(BytesIO(base64.b64decode(train_json_list[i]['imageData'])))
#     img = np.array(img, np.uint8)
#     title = []
#     for shape in train_json_list[i]['shapes']:
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, (0,255,0), 3)
#         title.append(shape['label'])
#     title = ','.join(title)
#     plt.imshow(img)
#     plt.subplot(6,5,i+1).set_title(title)
# plt.show()


# In[45]:


# 데이터셋
# PyTroch https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 를 활용하여 베이스라인을 작성하였습니다
# 튜토리얼 코드에 맞게 데이터셋을 정의합니다.


# In[14]:


class CustomDataset(Dataset):
    def __init__(self, train_json_list,train_jpg, mode= 'train'):
        self.mode = mode
        self.file_name = [json_file['imagePath'] for json_file in train_json_list] # ex)'sd_109.jpg'
        if mode == 'train':
            self.labels = []
            for data in train_json_list:
                label= []
                for shapes in data['shapes']:
                    label.append(shapes['label'])
                self.labels.append(label)
            self.points = []
            for data in train_json_list:
                point = []
                for shapes in data['shapes']:
                    point.append(shapes['points'])
                self.points.append(point)
        self.imgs = train_jpg
        self.widths = [1920 for i in range(len(train_jpg))]
        self.heights = [1080 for i in range(len(train_jpg))]
        
        self.label_map = {'sidewalk' : 1}
        
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.imgs)
        
        
    def __getitem__(self,i):
        file_name = self.file_name[i]
        img = Image.open(self.imgs[i]) #https://devlog.jwgo.kr/2019/11/04/how-to-convert-image-base64-both-way/
        img = self.transforms(img)
        
        target = {}
        if self.mode == 'train':
            boxes = []
            for point in self.points[i]:
                x_min = int(np.min(np.array(point)[:,0]))
                x_max = int(np.max(np.array(point)[:,0]))
                y_min = int(np.min(np.array(point)[:,1]))
                y_max = int(np.max(np.array(point)[:,1]))
                boxes.append([x_min, y_min, x_max, y_max])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
            area = (boxes[:,3]-boxes[:,1] * boxes[:,2]- boxes[:,0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            label = [self.label_map[label] for label in self.labels[i]]
            
            masks = []
            for box in boxes :
                mask = np.zeros([int(self.heights[i]), int(self.widths[i])], np.uint8)
                masks.append(cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 1, -1))
                
            masks = torch.tensor(masks, dtype=torch.uint8) # uint8 : 0~255?
            
            target['boxes'] = boxes
            target['labels'] = torch.tensor(label, dtype = torch.int64)
            target['masks']= masks
            target['iscrowd'] = iscrowd
        target['image_id'] = torch.tensor([i], dtype=torch.int64)
        if self.mode== 'test':
            target['file_name'] = file_name
        return img, target


# In[15]:


train_dataset = CustomDataset(train_json_list,train_jpg, mode='train')


# In[16]:


test_dataset = CustomDataset(test_json_list,test_jpg, mode='test')


# In[17]:


def collate_fn(batch):
    return tuple(zip(*batch))


# In[18]:


train_dataset = CustomDataset(train_json_list,train_jpg, mode='train')

torch.manual_seed(1)
indices = torch.randperm(len(train_dataset,)).tolist()
train_dataset = torch.utils.data.Subset(train_dataset, indices)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=collate_fn)


# In[163]:


# model

#Penn-Fudan Database for Pedestrian Detection and Segmentation로 학습된 Mask R-CNN 모델을 finetuning합니다.


# In[19]:


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# In[20]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# class 4 + background 1 = 5
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[21]:


# 필요 클래스 및 함수 정의


# In[22]:


# utils


# In[23]:


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
    
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# In[24]:


# engine


# In[25]:


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# In[26]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[27]:


# let's train it for 10 epochs
from torch.optim.lr_scheduler import StepLR
num_epochs = 1

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset


# In[28]:


torch.save(model.state_dict(), 'maskrcnn_weights.pth')
torch.save(model, 'maskrcnn_model.pth')


# In[29]:


# 추론 및 제출


# In[36]:


threshold = 0.8
results = {
     'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}

model.eval()

for img,target in tqdm(test_dataset):
    with torch.no_grad():
        prediction= model([img.to(device)])[0]
        
    idx = np.where(prediction['scores'].cpu().numpy() > threshold)[0]
    for i in idx:
        x_min, y_min, x_max, y_max = prediction['boxes'].cpu().numpy()[i]
        class_id = prediction['labels'].cpu().numpy()[i]
        confidence = prediction['scores'].cpu().numpy()[i]
        
        results['file_name'].append(target['file_name'])
        results['class_id'].append(class_id)
        results['confidence'].append(confidence)
        results['point1_x'].append(x_min)
        results['point1_y'].append(y_min)
        results['point2_x'].append(x_max)
        results['point2_y'].append(y_min)
        results['point3_x'].append(x_max)
        results['point3_y'].append(y_max)
        results['point4_x'].append(x_min)
        results['point4_y'].append(y_max)
        results['point4_y'].append(y_max)
        


# In[ ]:


submission = pd.DataFrame(results)
#submission.shape

submission.head()


# In[ ]:


# 시각화


# In[ ]:


plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


submission = submission[50:]
submission.reset_index(inplace=True,drop= True )

plt.figure(figsize=(25,20))

for j in submission.iterrows():
    
    plt.subplot(4,4,j[0]+1)
    
    fn = j[1][0]
    img = Image.open(f'./desktop/examples/semantic_segmentation/test_set/{fn}')
    img = np.array(img, np.uint8)
    
    
    title = []
    points = []
    points.append([j[1][3],j[1][4]])
    points.append([j[1][5],j[1][6]])
    points.append([j[1][7],j[1][8]])
    points.append([j[1][9],j[1][10]])
#     points.append([j['point1_x'],j['point1_y']])
#     points.append([j['point2_x'],j['point2_y']])
#     points.append([j['point3_x'],j['point3_y']])
#     points.append([j['point4_x'],j['point4_y']])
    

    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, (0,255,0), 3)
    title.append(j[1][0])
    #title = ','.join(title)
    plt.imshow(img)
    plt.subplot(4,4,j[0]+1).set_title(title)
    if j[0] == 15: break    
plt.show()


# In[ ]:


points


# In[ ]:


#img=Image.fromarray(img)
Image_sidewalk = img.crop((1400, 303, 1821, 503))
#croppedImage.save(os.path.join(saveTrafficWithLightDirName, fileNameWithoutExt + "-" + str(jndex + 1) + ".jpg"))


# In[ ]:


Image_sidewalk


# In[ ]:




