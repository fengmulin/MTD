import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
import pyclipper
import math
from concern.config import State
from .data_process import DataProcess2
# import numba
#@numba.jit(nopython=True)
#@numba.jit(nopython=False)
def draw_point(polygon,point,dis_x,dis_y,fill_mask):
    h,w = fill_mask.shape
    
    xmin = np.clip(int(np.floor(polygon[:, 0].min())),0, w-1)
    xmax = np.clip(int(np.ceil(polygon[:, 0].max())), 0 ,w-1)
    ymin =  np.clip(int(np.floor(polygon[:, 1].min())), 0 ,h-1)
    ymax =  np.clip(int(np.ceil(polygon[:, 1].max())), 0 ,h-1)
    if xmax>=dis_x.shape[1]:
        xmax=dis_x.shape[1]-1
    if ymax>=dis_x.shape[0]:
        ymax = dis_x.shape[0]-1
    width = (xmax - xmin + 1)
    height = (ymax - ymin + 1)
    #print(polygon)
    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin
    mask = np.zeros((height,width), dtype=np.float32)
    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
    inter_x = point[0]-xmin
    inter_y = point[1]-ymin
    
    cv2.fillPoly(mask, [np.array(polygon).astype(np.int32).reshape(-1, 2)], 1) 
    #print(mask[56][18])
    # print(polygon,inter_x,inter_y,point[1], point[0],2380-ymin,1242-xmin)
    # print(mask[2380-ymin,1242-xmin],fill_mask[ymin:ymin +height,xmin:xmin+width][2380-ymin,1242-xmin]
    #       ,(inter_x - xs)[2380-ymin,1242-xmin])
    # print((inter_x-xs).shape,mask.shape,fill_mask[ymin:ymin +height,xmin:xmin+width].shape)
    # print(fill_mask.shape,ymin,ymin +height,xmin,xmin+width)
    dis_xx = (inter_x - xs)*mask*fill_mask[ymin:ymin +height,xmin:xmin+width]
    dis_yy = (inter_y - ys)*mask*fill_mask[ymin:ymin +height,xmin:xmin+width]
    dis_x[ymin:ymin +height,xmin:xmin+width] +=dis_xx
    dis_y[ymin:ymin +height,xmin:xmin+width] +=dis_yy
    cv2.fillPoly(fill_mask[ymin:ymin +height,xmin:xmin+width], [np.array(polygon).astype(np.int32).reshape(-1, 2)], 0)
    return dis_x,dis_y,fill_mask
def draw_top(polygon,dis_x,dis_y,fill_mask,index1,index2):
    xmin = round(polygon[:, 0].min())
    xmax = round(polygon[:, 0].max())
    ymin = round(polygon[:, 1].min()) 
    ymax = round(polygon[:, 1].max())
    if xmax>=dis_x.shape[1]:
        xmax=dis_x.shape[1]-1
    if ymax>=dis_x.shape[0]:
        ymax = dis_x.shape[0]-1
    width = (xmax - xmin + 1)
    height = (ymax - ymin + 1)
    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin
    mask = np.zeros((height,width), dtype=np.float32)
    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
    # print(polygon)
    end1,end2 = polygon[index1], polygon[index2]
    # print(end1, end2)
    # raise
    if end1[0] == end2[0]:
        k=None
        b=ys
    else:
        k = (end2[1]-end1[1])/ (end2[0]-end1[0])
        b = end2[1] - k*end2[0]
    if k is None:
        chui_k =0
    elif k==0:
        chui_k =None
        chui_b = xs
    else:
        chui_k = -1/k
        chui_b = ys-chui_k*xs
    if k is None:
        inter_x = end1[0]
        inter_y = chui_k*inter_x + b
    elif chui_k is None:
        inter_x = xs
        inter_y = xs* k + b
    else:
        inter_x = (chui_b - b)/(k-chui_k)
        inter_y = k*inter_x + b
    cv2.fillPoly(mask, [np.array(polygon).astype(np.int32).reshape(-1, 2)], 1) 
    dis_xx = (inter_x - xs)*mask*fill_mask[ymin:ymin +height,xmin:xmin+width]
    dis_yy = (inter_y - ys)*mask*fill_mask[ymin:ymin +height,xmin:xmin+width]
    dis_x[ymin:ymin +height,xmin:xmin+width] +=dis_xx
    dis_y[ymin:ymin +height,xmin:xmin+width] +=dis_yy
    cv2.fillPoly(fill_mask[ymin:ymin +height,xmin:xmin+width], [np.array(polygon).astype(np.int32).reshape(-1, 2)], 0)
    return dis_x,dis_y,fill_mask

class MakeMagnetv2Map(DataProcess2):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=8)
    shrink_ratio = State(default=0.4)
    def __init__(self, **kwargs):
        self.load_all(**kwargs)
    def dist(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))     
    def midpoint(self, point1, point2):
        """计算两点之间的中点"""
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
    
    def four_equal_points(self, point1, point2):
        """生成两点之间的三个四等分点"""
        # 计算中点
        mid = self.midpoint(point1, point2)
        # 计算四等分点
        quarter1 = self.midpoint(point1, mid)
        quarter3 = self.midpoint(mid, point2)
        # quarter2 = midpoint(point1, quarter3)
        # quarter4 = midpoint(mid, quarter3)
        return [point1, quarter1, mid, quarter3, point2] 
    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']     
        h, w = image.shape[:2]

        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        dis_x = np.zeros((1,h,w), dtype=np.float32)
        dis_y = np.zeros((1,h,w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        fill_mask = np.ones((h,w))
        area =dict()
        polygonss=[]
        ins =0
        for i in range(len(polygons)):
            
            polygon = polygons[i]
            
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygonss.append(polygon)
                area[ins] = Polygon(polygon).area
                ins+=1
        sorted_items = sorted(area.items(), key=lambda x: x[1])  
        
        for key, value in sorted_items:
            polygon = polygonss[key]
            if polygon.shape[0]<10:
                polygon = np.array(self.four_equal_points(polygon[0], polygon[1])
                        +self.four_equal_points(polygon[2], polygon[3])).reshape(-1,2)
            lens = polygon.shape[0] // 2 
            point_a =[]
            point_b = []
            point_mid = []
            for iii in range(lens):
                
                point_a.append(polygon[iii]) 
                point_b.append(polygon[lens*2-1-iii])
                point_mid.append((polygon[iii] + polygon[lens*2-1-iii])/2)
            
            poly_temp = np.array([point_a[0], point_a[1], point_mid[1], point_mid[0]])
            dis_x[0],dis_y[0],fill_mask = draw_point(poly_temp,point_a[0],dis_x[0],dis_y[0],fill_mask)
            
            poly_temp = np.array([point_b[0], point_b[1], point_mid[1], point_mid[0]])
            dis_x[0],dis_y[0],fill_mask = draw_point(poly_temp,point_b[0],dis_x[0],dis_y[0],fill_mask)
            ### right edge

            poly_temp = np.array([point_a[lens-2], point_a[lens-1], point_mid[lens-1], point_mid[lens-2]])
            dis_x[0],dis_y[0],fill_mask = draw_point(poly_temp,point_a[lens-1],dis_x[0],dis_y[0],fill_mask)
            
            poly_temp = np.array([point_b[lens-2], point_b[lens-1], point_mid[lens-1], point_mid[lens-2]])
            dis_x[0],dis_y[0],fill_mask = draw_point(poly_temp,point_b[lens-1],dis_x[0],dis_y[0],fill_mask)
            
            for iii in range(1,lens-2):
                poly_temp = np.array([point_a[iii], point_a[iii+1], point_mid[iii+1], point_mid[iii]])
                dis_x[0],dis_y[0],fill_mask = draw_top(poly_temp,dis_x[0],dis_y[0],fill_mask,0,1)
                poly_temp = np.array([point_b[iii], point_b[iii+1], point_mid[iii+1], point_mid[iii]])
                dis_x[0],dis_y[0],fill_mask = draw_top(poly_temp,dis_x[0],dis_y[0],fill_mask,0,1)
                
        #print(kwmask.max(),kwmask.min())
        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt_x=dis_x, 
                    gt_y=dis_y,
                    # mask = mask,
                    dis_mask=1-fill_mask
                   )
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

