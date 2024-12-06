#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
import cv2

class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):

        def get_union_img(pD, pG):
            return np.sum(np.logical_or(pD, pG))

        def get_intersection_over_union_img(pD, pG):
            return get_intersection_img(pD, pG) / get_union_img(pD, pG)

        def get_intersection_img(pD, pG):
            return np.sum(np.logical_and(pD, pG))

        def get_union(pD, pG):
            # import Polygon as Polygonn
            # pD=Polygonn.Polygon(pD)
            # pG=Polygonn.Polygon(pG)
            # print(pD)
            # from shapely.geometry import Polygon
            areaA = Polygon(pD).area
            areaB = Polygon(pG).area
            return areaA + areaB - get_intersection(pD, pG)

        def get_intersection_over_union(pD, pG):

            # print(get_intersection(pD, pG),get_union(pD, pG))
            return get_intersection(pD, pG) / get_union(pD, pG);


        # def get_intersection(pD, pG):
        #     return Polygon(pD).intersection(Polygon(pG)).area
        
        def get_intersection(pD, pG):
            import Polygon as Polygonn
            pD=Polygonn.Polygon(pD)
            pG=Polygonn.Polygon(pG)
            pInt = pD & pG
            if len(pInt) == 0:
                return 0
            return pInt.area()
            # return Polygon(pD).intersection(Polygon(pG)).area
        
        def box_to_img(box1, box2):
            # print(box1.shape, box2.shape)
            xmin = np.floor(min(np.min(box1[:, 0]), np.min(box2[:, 0]))).astype(np.int)
            xmax = np.ceil(max(np.max(box1[:, 0]), np.max(box2[:, 0]))).astype(np.int)
            ymin = np.floor(min(np.min(box1[:, 1]), np.min(box2[:, 1]))).astype(np.int)
            ymax = np.ceil(max(np.min(box1[:, 1]), np.max(box2[:, 1]))).astype(np.int)

            mask1 = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            mask2 = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            box1[:, 0] = box1[:, 0] - xmin
            box1[:, 1] = box1[:, 1] - ymin
            box2[:, 0] = box2[:, 0] - xmin
            box2[:, 1] = box2[:, 1] - ymin
            cv2.fillPoly(mask1, box1.reshape(1, -1, 2).astype(np.int32), 1)
            cv2.fillPoly(mask2, box2.reshape(1, -1, 2).astype(np.int32), 1)
            return mask1,mask2
        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct)/(n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []
        tp = []
        fp = []
        fn = []
        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']
            if len(points)<4:
                continue
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                # print(points)
                continue
            fn.append(len(gtPols))
            gtPol = points
            gtPols.append(gtPol)
            # print('append',)
            gtPolPoints.append(points)
            if dontCare:
                fn.remove(len(gtPols)-1)
                gtDontCarePolsNum.append(len(gtPols)-1)
        
        # raise
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(
            gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")
        
        for n in range(len(pred)):
            points = pred[n]['points']
            if len(points)<4:
                continue
            # qzz = Polygon(points)
            # print(len(qzz))
            # from shapely.ops import unary_union
            # from shapely.geometry import MultiPolygon
            # if not qzz.is_valid:
            #     simplified_polygon = qzz.simplify(0.1, preserve_topology=True)
            #     points = np.array(simplified_polygon.exterior.coords)

                # epsilon = 0.004 * cv2.arcLength(np.array(points).reshape(-1,2), True)
                # approx = cv2.approxPolyDP(np.array(points).reshape(-1,2), epsilon, True)
                # points = approx.reshape((-1, 2))
                # if not Polygon(points).is_valid or not Polygon(points).is_simple:
                #     epsilon = 0.004 * cv2.arcLength(np.array(points).reshape(-1,2), True)
                #     approx = cv2.approxPolyDP(np.array(points).reshape(-1,2), epsilon, True)
                #     points = approx.reshape((-1, 2))
                #     # continue
                # # points = pointss
                # if points.shape[0]<4:
                #     print('qzz')
                #     continue
                
                # if not Polygon(points).is_valid or not Polygon(points).is_simple:
                #     print('qzzzz')
                #     continue
                # print(points,33)
                # qzz = qzz.buffer(0)
                # if isinstance(qzz, Polygon):
                #     points = np.array(qzz.exterior)
                # else:
                #     epsilon = 0.002 * cv2.arcLength(np.array(points).reshape(-1,2), True)
                #     approx = cv2.approxPolyDP(np.array(points).reshape(-1,2), epsilon, True)
                #     points = approx.reshape((-1, 2))
                #     # if not Polygon(points).is_valid or not Polygon(points).is_simple:
                #     #     continue
                
                # try: 
                #     if not Polygon(points).is_valid or not Polygon(points).is_simple:
                #         continue
                #         # print(points,11)
                #         # epsilon = 0.001 * cv2.arcLength(points, True)
                #         # approx = cv2.approxPolyDP(points, epsilon, True)
                #         # points = approx.reshape((-1, 2))
                #         # print(points)
                #         # if not Polygon(points).is_valid or not Polygon(points).is_simple:
                #         #     continue
                #     # print(points, 44)
                # except:
                #     # print(points,22)
                #     continue
            fp.append(len(detPols))
            # print('append', len(detPols))
            #     # print(points)
            #     continue
            #     points = np.array(Polygon(points).buffer(0).exterior).reshape(-1,2)
                # epsilon = 0.006 * cv2.arcLength(np.array(points).reshape(-1,1,2), True)
                # approx = cv2.approxPolyDP(np.array(points).reshape(-1,1,2), epsilon, True)
                # points = approx.reshape((-1, 2))
                # continue
            # if not Polygon(points).is_valid:
            #     continue
            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    # print(detPol, dontCarePol )
                    intersected_area = get_intersection(dontCarePol, detPol)
                    # polygon1 = np.array(dontCarePol).reshape(-1,2).astype(np.int32)
                    # polygon2 = np.array(detPol).reshape(-1,2).astype(np.int32)
                    # img1,img2 = box_to_img(polygon1, polygon2)
                    # intersected_area = get_intersection_img(img1, img2)
                    # if intersected_area!=0:
                        
                    #     print(intersected_area1, intersected_area)
                    # if Polygon(dontCarePol).is_valid and Polygon(detPol).is_valid:
                    #     intersected_area = get_intersection(dontCarePol, detPol)
                    # else:
                        
                    #     polygon1 = np.array(dontCarePol).reshape(-1,2).astype(np.int32)
                    #     polygon2 = np.array(detPol).reshape(-1,2).astype(np.int32)
                    #     img1,img2 = box_to_img(polygon1, polygon2)
                    #     # max_x = int(max(np.max(polygon1[:, 0]), np.max(polygon2[:, 0])))
                    #     # max_y = int(max(np.max(polygon1[:, 1]), np.max(polygon2[:, 1])))

                    #     # img1 = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
                    #     # img2 = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
                    #     # cv2.fillPoly(img1, [polygon1], 1)
                    #     # cv2.fillPoly(img2, [polygon2], 1)
                    #     intersected_area = get_intersection_img(img1, img2)
                    # intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    # print(precision, 888888888)
                    if (precision > self.area_precision_constraint):
                        
                        detDontCarePolsNum.append(len(detPols)-1)
                        fp.remove(len(detPols)-1)
                        # print('dontcare', n)
                        break
        
        
        # print(len(pred), len(detPols))
        evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(
            detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                    # polygon1 = np.array(pG).reshape(-1,2).astype(np.int32)
                    # polygon2 = np.array(pD).reshape(-1,2).astype(np.int32)

                    # img1,img2 = box_to_img(polygon1, polygon2)
                    # iouMat[gtNum, detNum] = get_intersection_over_union_img(img1, img2)
                    # if Polygon(pG).is_valid and Polygon(pD).is_valid:
                    #     iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                    # else:
                    #     polygon1 = np.array(pG).reshape(-1,2).astype(np.int32)
                    #     polygon2 = np.array(pD).reshape(-1,2).astype(np.int32)

                    #     img1,img2 = box_to_img(polygon1, polygon2)
                    #     # max_x = int(max(np.max(polygon1[:, 0]), np.max(polygon2[:, 0])))
                    #     # max_y = int(max(np.max(polygon1[:, 1]), np.max(polygon2[:, 1])))

                    #     # img1 = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
                    #     # img2 = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
                    #     # cv2.fillPoly(img1, [polygon1], 1)
                    #     # cv2.fillPoly(img2, [polygon2], 1)
                        
                    #     iouMat[gtNum, detNum] = get_intersection_over_union_img(img1, img2)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            tp.append(detNum)
                            fn.remove(gtNum)
                            # print('remove',detNum)
                            fp.remove(detNum)
                            
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        # print(len(detPols), len(detDontCarePolsNum), 8885)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
            precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched,
            'evaluationLog': evaluationLog,
            'tp':tp,
            'fp':fp,
            'fn':fn
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']
        # print(numGlobalCareGt, matchedSum)
        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum)/numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum)/numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
            methodRecall * methodPrecision / (methodRecall + methodPrecision)

        methodMetrics = {'precision': methodPrecision,
                         'recall': methodRecall, 'hmean': methodHmean}

        return methodMetrics


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
