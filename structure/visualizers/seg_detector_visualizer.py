import cv2
import concern.webcv2 as webcv2
import numpy as np
import torch

from concern.config import Configurable, State
from data.processes.make_icdar_data import MakeICDARData


class SegDetectorVisualizer(Configurable):
    vis_num = State(default=4)
    eager_show = State(default=False)

    def __init__(self, **kwargs):
        cmd = kwargs['cmd']
        if 'eager_show' in cmd:
            self.eager_show = cmd['eager_show']

    def visualize(self, batch, output_pair, pred, metric):
        #print(batch['filename'])
        boxes, _ = output_pair
        result_dict = {}
        for i in range(batch['image'].size(0)):
            result_dict.update(
                self.single_visualize(batch, i, boxes[i], pred, metric))
        if self.eager_show:
            #print("*******")
            webcv2.waitKey()
            return {}
        return result_dict

    def _visualize_heatmap(self, heatmap, canvas=None):
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap[0] * 255).astype(np.uint8)
        if canvas is None:
            pred_image = heatmap
        else:
            pred_image = (heatmap.reshape(
                *heatmap.shape[:2], 1).astype(np.float32) / 255 + 1) / 2 * canvas
            pred_image = pred_image.astype(np.uint8)
        return pred_image
    def _visualize_heatmap_regression(self, heatmap, canvas=None):
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        # print(heatmap.max(),heatmap.min(),22)
        #heatmap = (heatmap[0] * 255).astype(np.uint8)
        if canvas is None:
            # print(heatmap[0].sum(),heatmap[0].shape)
            # pred_image = ((heatmap[0])*20).astype(np.uint8)
            print(heatmap[0].max())
            pred_image = ((heatmap[0]-heatmap[0].min())/(heatmap[0].max()-heatmap[0].min())*255).astype(np.uint8)
        else:
            pred_image = (heatmap[0].reshape(
                *heatmap.shape[:2], 1).astype(np.float32) / 255 + 1) / 2 * canvas
            pred_image = pred_image.astype(np.uint8)
        
        return pred_image


    def single_visualize(self, batch, index, boxes, pred, metrics):
        image = batch['image'][index]
        #print(image[:,0,0],2)
        #raise
        polygons = batch['polygons'][index]
        if isinstance(polygons, torch.Tensor):
            polygons = polygons.cpu().data.numpy()
        ignore_tags = batch['ignore_tags'][index]
        original_shape = batch['shape'][index]
        filename = batch['filename'][index]
        image =image.cpu().numpy()
        # std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        # mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        #image = (image * std + mean).transpose(1, 2, 0) * 255
        image = (image).transpose(1, 2, 0) * 255
        
        pred_canvas = image.copy()
        
        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        pred_canvas += RGB_MEAN
        pred_canvas[pred_canvas<0] = 0
        # print(pred_canvas.max(),pred_canvas.min())
        # print(pred_canvas[0])
        # raise
        pred_canvas = pred_canvas.astype(np.uint8)
        
        #print(pred_canvas.shape)
        pred_canvas = cv2.resize(pred_canvas, (int(original_shape[1]), int(original_shape[0])))
        pred_canvas2 = pred_canvas.copy() 
        if isinstance(pred, dict) and 'gap' in pred:
            gap = self._visualize_heatmap(pred['gap'][index])
        # print(pred.shape)
        # raise   
        if isinstance(pred, dict) and 'binary' in pred:
            binary = self._visualize_heatmap(pred['binary'][index])
        else:
            binary = self._visualize_heatmap(pred[index])
        #binary = self._visualize_heatmap(pred[index])
         
        
        #MakeICDARData.polylines(self, binary, polygons, ignore_tags)
        
        if isinstance(pred, dict) and 'ori_binary' in pred:
            ori_binary = self._visualize_heatmap(pred['ori_binary'][index])
        if isinstance(pred, dict) and 'gauss' in pred:
            gauss = self._visualize_heatmap_regression(pred['gauss'][index])
        if isinstance(pred, dict) and 'area' in pred:
            #print(pred['area'][index].shape,11)
            area = self._visualize_heatmap_regression(pred['area'][index])
        if isinstance(pred, dict) and 'pos' in pred:
            #print(pred['pos'][index].shape)
            pos = self._visualize_heatmap_regression(pred['pos'][index].unsqueeze(0))
            #print(pos.shape)
        #     raise
            # print(area.max(), area.min())
        # if isinstance(pred, dict) and 'pos' in pred:
        #     print(pos.shape)
        #     pos = self._visualize_heatmap(pred['pos'][index])
        #     print(pos.shape)
        #     raise
        #    # MakeICDARData.polylines(self, ori_binary, polygons, ignore_tags)

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 3)
            
        for ins in metrics['tp']:
            # print(ins,len(metrics['detPolPoints']))
            box = np.array(metrics['detPolPoints'][ins]).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas2, [box], True, (0, 255, 0), 3)
        for ins in metrics['fp']:
            box = np.array(metrics['detPolPoints'][ins]).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas2, [box], True, (255, 0, 0), 3)
        for ins in metrics['fn']:
            box = np.array(metrics['gtPolPoints'][ins]).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas2, [box], True, (0, 0, 255), 3)
        for ins in metrics['gtDontCare']:
            box = np.array(metrics['gtPolPoints'][ins]).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas2, [box], True, (0, 255, 255), 2)
            # if isinstance(pred, dict) and 'ori_binary' in pred:
            #     cv2.polylines(ori_binary, [box], True, (0, 255, 0), 1) 

        if self.eager_show:
            webcv2.imshow(filename + ' output', cv2.resize(pred_canvas, (1024, 1024)))
            if isinstance(pred, dict) and 'gap' in pred:
                webcv2.imshow(filename + ' gap', cv2.resize(gap, (1024, 1024)))
                webcv2.imshow(filename + ' pred', cv2.resize(pred_canvas, (1024, 1024)))
            if isinstance(pred, dict) and 'ori_binary' in pred:
                webcv2.imshow(filename + ' ori_binary', cv2.resize(ori_binary, (1024, 1024)))
            return {}
        else:
            # print('qzz')
            if isinstance(pred, dict) and 'gauss' in pred:
                #print("quan")
                return {
                    filename + '_output': pred_canvas,
                    filename + '_binary': binary,
                    filename + '_gauss': gauss,
                    # filename + '_ori': ori_binary,
                    # filename + '_pos0': pos[0],
                    # filename + '_pos1': pos[1],
                    # filename + '_pos2': pos[2],
                    # filename + '_pos3': pos[3]
                    
                }
            else:
                return {
                filename + '_output': pred_canvas,
                filename + '_binary': binary,
                filename + '_judge': pred_canvas2,
                # filename + '_ori': ori_binary
            }

    def demo_visualize(self, image_path, output):
        boxes, _ = output
        boxes = boxes[0]
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_shape = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)

        return pred_canvas

