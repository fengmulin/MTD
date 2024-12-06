import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
import math
from concern.config import State
from .data_process import DataProcess2


class MakeFBShrinkMap(DataProcess2):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=8)
    sr = State(default=0.25)
    area_middle = State(default=4369)
    
    def __init__(self, **kwargs):
        self.load_all(**kwargs)
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
    def dist(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2)) 
    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        # mid_area = 0
        # ints = 0
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']
        
        h, w = image.shape[:2]
        # if data['filename'] =='../../dataset/total_text//train_images/img949.jpg':
        #         print(polygon,11)
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        # kwmask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            
            polygon = polygons[i]
            
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])

            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # print(polygon)
                if polygon.shape[0]%2==1:
                    cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                if polygon.shape[0] == 4:
                    if self.dist(polygon[0],polygon[1])<self.dist(polygon[1],polygon[2]):
                        polygon[[0,1,2,3], :] = polygon[[1,2,3,0], :]
                        # print('qzzzzzzzzzz')
                    # else:
                    #     print('aaaaaaaaaaa')
                # if polygon.shape[0]==4:
                    
                    polygon = np.array(self.four_equal_points(polygon[0], polygon[1])
                                    +self.four_equal_points(polygon[2], polygon[3])).reshape(-1,2)
                # elif polygon.shape[0]<10:
                #     len_polygon = polygon.shape[0]
                #     err = 
                #     print(len_polygon,'qzzzzzz',polygon)
                #     polygon = np.array(self.four_equal_points(polygon[0], polygon[len_polygon//2-1])
                #                     +self.four_equal_points(polygon[len_polygon//2], polygon[len_polygon-1])).reshape(-1,2)
                dis = []
                poly_shrink = []
                lens = polygon.shape[0]
                qzz = lens//2-1
                
                for iii in range(lens//2):
                    dis.append([polygon[iii][0]-polygon[lens-1-iii][0], polygon[iii][1]-polygon[lens-1-iii][1]])
                # for iii in range(lens//2-1):
                #     dis.append([polygon[iii][0]-polygon[lens-1-iii][0], polygon[iii][1]-polygon[lens-1-iii][1]])
                poly_shrink.append([(polygon[0][0]-self.sr*dis[0][0]+polygon[1][0]-self.sr*dis[1][0])/2, 
                                    (polygon[0][1]-self.sr*dis[0][1]+polygon[1][1]-self.sr*dis[1][1])/2])
                
                for iii in range(1,qzz): 
                    poly_shrink.append([polygon[iii][0]-self.sr*dis[iii][0], polygon[iii][1]-self.sr*dis[iii][1]])
                    # print(iii)
                poly_shrink.append([(polygon[qzz-1][0]-self.sr*dis[qzz-1][0]+polygon[qzz][0]-self.sr*dis[qzz][0])/2, 
                                    (polygon[qzz-1][1]-self.sr*dis[qzz-1][1]+polygon[qzz][1]-self.sr*dis[qzz][1])/2])
                # print(qzz)
                try:
                    
                    poly_shrink.append([(polygon[qzz+1][0]+self.sr*dis[lens-qzz-2][0]+polygon[qzz+2][0]+self.sr*dis[lens-qzz-3][0])/2, 
                                    (polygon[qzz+1][1]+self.sr*dis[lens-qzz-2][1]+polygon[qzz+2][1]+self.sr*dis[lens-qzz-3][1])/2])
                except:
                    print(polygon, lens, qzz)
                    raise
                # print(qzz+1)
                for iii in range(lens//2+1, lens-1):
                    # print(iii)
                    poly_shrink.append([polygon[iii][0]+self.sr*dis[lens-1-iii][0],
                                        polygon[iii][1]+self.sr*dis[lens-1-iii][1]])
                poly_shrink.append([(polygon[lens-1][0]+self.sr*dis[0][0]+polygon[lens-2][0]+self.sr*dis[1][0])/2, 
                                    (polygon[lens-1][1]+self.sr*dis[0][1]+polygon[lens-2][1]+self.sr*dis[1][1])/2])
                shrinked = np.array(poly_shrink).astype(np.int32).reshape(-1,2)
                
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)
                # raise
        # print(mask.shape)
        # #print(kwmask.max(),kwmask.min())
        # cv2.imwrite('xx.png',mask*125)
        # print(filename)
        # raise
        # # cv2.imwrite('yy.png',(w_mask-gt[0])*255)
        # cv2.imwrite('vis/'+filename.split('/')[-1]+'x.jpg',(gt[0])*255)
        # cv2.imwrite('vis/'+filename.split('/')[-1]+'y.jpg',image)
        # raise
        # print(gt.dtype)
        # raise
        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask
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

