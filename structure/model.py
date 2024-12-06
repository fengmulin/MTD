import os
from tkinter import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        #self.to(self.device)
        self.criterion.cuda()
        self.model.cuda()

    @staticmethod
    def model_name(args):
        return  args['backbone'] + '^' +args['decoder']+"^" +args['loss_class']
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True,visualize =False,speed=False,times =None):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
       #
            #data = Variable(batch).cuda()
            #data = batch
            #data = batch.cuda()
            
        # for i in  batch:
        #     print(i)
        # raise
        data = data.float()
        # if not training:
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     self.model.eval()
        #     self.model.to(device)
        #try:
        if speed:
            import time
            start = time.time()
            for _ in range(times):
                pred = self.model(data, training=self.training)  
            end = time.time()
            return pred,(end-start)/times    
        pred = self.model(data, training=self.training)
        # except:
        #     print(batch['filename'])
        
        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred