import cv2
import numpy as np
from shapely.geometry import Polygon,LineString
import pyclipper
from concern.config import Configurable, State
# import numba
# @numba.jit(nopython=True)
def cal_box(coordinates, x_offset_map, y_offset_map, width_ratio,height_ratio):
    
    height, width = x_offset_map.shape

    # 限制坐标范围到 [0, width-1] 和 [0, height-1]
    limited_coordinates = np.round(np.clip(coordinates, [0, 0], [width - 1, height - 1])).astype(int)
    
    x_coords, y_coords = limited_coordinates[:, 0], limited_coordinates[:, 1]

    # print(height, width)
    # print(y_coords, x_coords)
    x_offsets = x_offset_map[y_coords, x_coords]
    y_offsets = y_offset_map[y_coords, x_coords]

    box = coordinates + np.stack([x_offsets, y_offsets], axis=1)
    box[:, 0] = box[:, 0]* width_ratio
    box[:, 1] = box[:, 1]* height_ratio
    return box

# def cal_box(coordinates, x_offset_map, y_offset_map, width_ratio,height_ratio):
#     """
#     计算轮廓偏移后的坐标。
    
#     参数：
#     - coordinates: k*2 的 numpy 数组，代表 k 个点的坐标。
#     - x_offset_map: 640*640 的 numpy 数组，横轴偏移图。
#     - y_offset_map: 640*640 的 numpy 数组，纵轴偏移图。
    
#     返回值：
#     - 偏移后的坐标，大小为 k*2 的 numpy 数组。
#     """
#     # 确保输入的坐标是整数（像素索引）
#     pixel_indices = np.round(coordinates).astype(int)
    
#     # 提取 x 和 y 坐标
#     x_coords, y_coords = pixel_indices[:, 0], pixel_indices[:, 1]
#     x_offsets = x_offset_map[y_coords, x_coords]
#     y_offsets = y_offset_map[y_coords, x_coords]
#     # 获取偏移值
#     box = coordinates + np.stack([x_offsets, y_offsets], axis=1)
#     box[:, 0] = box[:, 0]* width_ratio
#     box[:, 1] = box[:, 1]* height_ratio
#     return box
class SegDetectorRepresenter(Configurable):
    thresh = State(default=0.3)
    box_thresh = State(default=0.7)
    max_candidates = State(default=100)
    dest = State(default='binary')

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.min_size = 4
        #self.scale_ratio = 0.4
        if 'debug' in cmd:
            self.debug = cmd['debug']
        if 'thresh' in cmd:
            self.thresh = cmd['thresh']
        if 'box_thresh' in cmd:
            self.box_thresh = cmd['box_thresh']
        if 'dest' in cmd:
            self.dest = cmd['dest']
        #self.box_thresh =0.5
        # width, height, binary, dis_x, dis_y, is_output_polygon=self.args['polygon']
    def represent(self, batch, pred, height, width, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, 1, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, 1, H, W)
        '''

        binary,dis_x,dis_y = pred[0],pred[1],pred[2]
        segmentation = self.binarize(binary)
        boxes_batch = []
        scores_batch = []
        # print(batch['shape'])
        # raise
        # height, width = batch['shape'][0].numpy()
        
        if is_output_polygon:
            boxes, scores = self.polygons_from_bitmap(
                binary,dis_x,dis_y,
                segmentation, width, height)
        else:
            boxes, scores = self.boxes_from_bitmap(
                binary,dis_x,dis_y,
                segmentation, width, height)
    
        boxes_batch.append(boxes)
        scores_batch.append(scores)
       
        return boxes_batch, scores_batch
    
    def binarize(self, pred):
        #return pred > 0.3
        return pred > self.thresh

    def polygons_from_bitmap(self,binary,dis_x,dis_y,  bitmap,dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = bitmap 
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap).astype(np.uint8),
            cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            ## 精简转换
            epsilon = 0.004 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0]<4:
                continue
            ## 直接转换
            # points = contour.reshape(-1,2)
            score = self.box_score_fast(binary, points)
            if self.box_thresh > score:
                continue
            box=[]
            width_ratio = dest_width/width
            height_ratio = dest_height/height
            box = cal_box(points,dis_x ,dis_y,width_ratio,height_ratio)
            box = np.round(box)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, binary,dis_x,dis_y,  bitmap,dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        # print(_bitmap.size)
        # raise
        # assert _bitmap.size(0) == 1  # The first channel
        #pred = pred
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        for index in range(num_contours):
            contour = contours[index]
            bounding_box = cv2.minAreaRect(contour)
            sside = min(bounding_box[1])
            if sside < self.min_size:
                continue
            bbox = cv2.boxPoints(bounding_box)
            score = self.box_score_fast(binary, bbox)
            if self.box_thresh > score:
                continue
            points = np.array(bbox)

            # t4 = time.time()
            width_ratio = dest_width/width
            height_ratio = dest_height/height
        #    # pointss = np.round(points)
            # print('qzzf')
            # raise
            box = cal_box(points,dis_x ,dis_y,width_ratio,height_ratio)
            box = np.round(box)
            boxes[index, :, :] = box
            scores[index] = score
        return boxes, scores

    def get_mini_boxes(self, contour):
        contour = np.float32(contour)
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
