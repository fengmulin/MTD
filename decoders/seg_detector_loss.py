import sys
from decoders.loss import l1_loss

import torch
import torch.nn as nn
from .loss.dice_loss import DiceLoss
from .loss.l1_loss import MaskL1Loss
from .loss.lnl1_loss import MasklnL1Loss
from .loss.lnl1_loss import Masklog2L1Loss
from .loss.regression_loss import RegLoss
from .loss.balance_cross_entropy_loss import BalanceCrossEntropyLoss
from .loss.balance_cross_entropy_loss_weight import BalanceCrossEntropyLoss_weight

class SegDetectorLossBuilder():
    '''
    Build loss functions for SegDetector.
    Details about the built functions:
        Input:
            pred: A dict which contains predictions.
                thresh: The threshold prediction
                binary: The text segmentation prediction.
                thresh_binary: Value produced by `step_function(binary - thresh)`.
            batch:
                gt: Text regions bitmap gt.
                mask: Ignore mask,
                    pexels where value is 1 indicates no contribution to loss.
                thresh_mask: Mask indicates regions cared by thresh supervision.
                thresh_map: Threshold gt.
        Return:
            (loss, metrics).
            loss: A scalar loss value.
            metrics: A dict contraining partial loss values.
    '''

    def __init__(self, loss_class, *args, **kwargs):
        self.loss_class = loss_class
        self.loss_args = args
        self.loss_kwargs = kwargs

    def build(self):
        return getattr(sys.modules[__name__], self.loss_class)(*self.loss_args, **self.loss_kwargs)


class BaseDiceLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(BaseDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, pred, batch):

        dice_loss = self.dice_loss(pred['binary'], batch['gt'], batch['mask'])
        loss =  dice_loss 
        metrics = dict(dice_loss=dice_loss)
      
        return loss, metrics
class BaseLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(BaseLoss, self).__init__()
        self.bce_loss = BalanceCrossEntropyLoss()

    def forward(self, pred, batch):

        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        loss =  bce_loss 
        metrics = dict(bce_loss=bce_loss)
      
        return loss, metrics

class BaseLoss2(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(BaseLoss2, self).__init__()

        self.bce_loss = BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()

    def forward(self, pred, batch):

        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        ori_loss = self.bce_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        
        loss = ori_loss  + bce_loss 
        metrics = dict(bce_loss=bce_loss)
        metrics['roi_loss'] = ori_loss
        return loss, metrics
    
class DisFocusLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DisFocusLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = MaskL1Loss()
        self.disy_loss = MaskL1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()
        self.area_scale = area_scale
        self.pos_scale = pos_scale

    def forward(self, pred, batch):


        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['ori_gt'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['ori_gt'])

        loss =  bce_loss *6 + disx_loss*0.1+disy_loss*0.1
        metrics = dict(bce_loss=bce_loss)

        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss

        return loss, metrics  



class DiceDisOriFocusLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DiceDisOriFocusLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.dice_loss2 = DiceLoss(eps=eps)
        self.disx_loss = MaskL1Loss()
        self.disy_loss = MaskL1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()
        self.area_scale = area_scale
        self.pos_scale = pos_scale

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        dice_loss = self.dice_loss2(pred['binary'], batch['gt'], batch['mask'])
        ori_loss = self.dice_loss (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['ori_gt'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['ori_gt'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  dice_loss *6 + disx_loss*0.1+disy_loss*0.1+3*ori_loss
        metrics = dict(dice_loss=dice_loss)
        #metrics['bce_loss2'] = bce_loss2
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        metrics['ori_loss'] = ori_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics 

class DisOriFocusLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DisOriFocusLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = MaskL1Loss()
        self.disy_loss = MaskL1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()
        self.area_scale = area_scale
        self.pos_scale = pos_scale

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        ori_loss = self.dice_loss (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['ori_gt'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['ori_gt'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  bce_loss *6 + disx_loss*0.1+disy_loss*0.1+3*ori_loss
        metrics = dict(bce_loss=bce_loss)
        #metrics['bce_loss2'] = bce_loss2
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        metrics['ori_loss'] = ori_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics     
class DisLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DisLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = MaskL1Loss()
        self.disy_loss = MaskL1Loss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['dis_mask'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['dis_mask'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  bce_loss *6 + disx_loss*0.1+disy_loss*0.1
        metrics = dict(bce_loss=bce_loss)
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics 
    
class DisOriMSELoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DisOriMSELoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = MaskL1Loss()
        self.disy_loss = MaskL1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = torch.nn.MSELoss()
        self.reg_loss2 = torch.nn.MSELoss()

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        ori_loss = self.dice_loss (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        disx_loss = self.reg_loss(pred['dis_x'][:,0,:,:]*batch['dis_mask'], batch['gt_x'][:,0,:,:]*batch['dis_mask'])
        disy_loss = self.reg_loss2(pred['dis_y'][:,0,:,:]*batch['dis_mask'], batch['gt_y'][:,0,:,:]*batch['dis_mask'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  bce_loss *6 + disx_loss*0.1+disy_loss*0.1+3*ori_loss
        metrics = dict(bce_loss=bce_loss)
        #metrics['bce_loss2'] = bce_loss2
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        metrics['ori_loss'] = ori_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics 
    
class DisOriLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DisOriLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = MaskL1Loss()
        self.disy_loss = MaskL1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        ori_loss = self.dice_loss (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['dis_mask'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['dis_mask'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  bce_loss *6 + disx_loss*0.1+disy_loss*0.1+3*ori_loss
        metrics = dict(bce_loss=bce_loss)
        #metrics['bce_loss2'] = bce_loss2
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        metrics['ori_loss'] = ori_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics 
class DisOrilnLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DisOrilnLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = MasklnL1Loss()
        self.disy_loss = MasklnL1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()
        self.area_scale = area_scale
        self.pos_scale = pos_scale

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        #print(pred['pos'].shape, batch['gt_pos'].shape,333)
        # print(pred['dis_x'].shape, batch['gt_x'].shape)
        # raise
        ori_loss = self.bce_loss (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['dis_mask'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['dis_mask'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  bce_loss *6 + disx_loss*0.2+disy_loss*0.2+3*ori_loss
        metrics = dict(bce_loss=bce_loss)
        #metrics['bce_loss2'] = bce_loss2
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        metrics['ori_loss'] = ori_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics 
    
class DislnLoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, area_scale=0.5, pos_scale=1.):
        super(DislnLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.disx_loss = Masklog2L1Loss()
        self.disy_loss = Masklog2L1Loss()
        self.bce_loss =  BalanceCrossEntropyLoss_weight()
        self.bce_loss2 = BalanceCrossEntropyLoss()
        self.reg_loss = RegLoss()
        self.reg_loss2 = RegLoss()
        self.area_scale = area_scale
        self.pos_scale = pos_scale

    def forward(self, pred, batch):

        #bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],batch['kwmask']) 
        bce_loss = self.bce_loss2(pred['binary'], batch['gt'], batch['mask'])
        #print(pred['pos'].shape, batch['gt_pos'].shape,333)
        # print(pred['dis_x'].shape, batch['gt_x'].shape)
        # raise
        disx_loss = self.disx_loss(pred['dis_x'], batch['gt_x'], batch['dis_mask'])
        disy_loss = self.disy_loss(pred['dis_y'], batch['gt_y'], batch['dis_mask'])
        #ori_loss = self.dice_loss2 (pred['ori_binary'], batch['ori_gt'], batch['ori_mask'])
        loss =  bce_loss *6 + disx_loss*0.5+disy_loss*0.5
        metrics = dict(bce_loss=bce_loss)
        #metrics['bce_loss2'] = bce_loss2
        metrics['diy_loss'] = disy_loss
        metrics['dix_loss'] = disx_loss
        #metrics['area_loss'] = area_loss
        return loss, metrics 

  