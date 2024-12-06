import os
#from tkinter import Variable
import cv2
import torch
import shutil
from tqdm import tqdm
from torch.autograd import Variable
from experiment import Experiment
from data.data_loader import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from data.image_dataset import ImageDataset
from data.data_loader import DataLoader
from concern.config import Configurable, Config

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, max_len=50):
        self.val = []
        self.count = []
        self.max_len = max_len
        self.avg = 0

    def update(self, val, n=1):
        self.val.append(val * n)
        self.count.append(n)
        if self.max_len > 0 and len(self.val) > self.max_len:
            self.val = self.val[-self.max_len:]
            self.count = self.count[-self.max_len:]
        self.avg = sum(self.val) / sum(self.count)
class Trainer:
    def __init__(self, experiment: Experiment,args):
        self.init_device() 
        self.args = args
        self.save_before_val = True
        self.experiment = experiment
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver
        self.f1 = 0
        self.best_recall = 0
        self.best_pre = 0
        self.best_epoch = 0
        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.work_dir = os.getcwd()
        # if self.args.get('finetune_iter',False):
        self.val_best =0
        self.total = 0
        
        ### loss 
        self.losses = AverageMeter()
        self.losses_item = dict()
        self.losses_item['bce_loss'] = AverageMeter()
        self.losses_item['size_loss'] = AverageMeter()
        self.losses_item['ori_loss'] = AverageMeter()
        self.losses_item['gauss_loss'] = AverageMeter()
        self.losses_item['tiny_loss'] = AverageMeter()
        self.losses_item['dix_loss'] = AverageMeter()
        self.losses_item['diy_loss'] = AverageMeter()
        self.losses_item['dice_loss'] = AverageMeter()


    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)
        return model

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def train(self):
    
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        model = self.init_model()
        model = model.cuda()
        self.copy_cur_env(self.work_dir, self.model_saver.dir_path+'DB')
        train_data_loader = self.experiment.train.data_loader
        if self.experiment.validation:
            validation_loaders = self.experiment.validation.data_loaders

        self.steps = 0
        optimizer = self.experiment.train.scheduler.create_optimizer(
            model.parameters())
        self.total = len(train_data_loader)
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            self.experiment.train.checkpoint.recover_model(
                model, self.device, self.logger,optimizer)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta
        
        self.logger.report_time('Init')
        model.train()
        
        while True:
            #print(self.steps)
            #self.validate(validation_loaders, model.model.module, epoch, self.steps)
            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            
            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)
                if self.args.get('finetune_iter',False):
                    if self.val_best>0:
                        self.val_best -= 1
                self.logger.report_time("Data loading")
                
                if self.logger.verbose:
                    torch.cuda.synchronize()
                self.train_step(model, optimizer, batch,
                                epoch=epoch, step=self.steps)

                self.logger.report_time('Forwarding ')
                self.steps += 1
                self.logger.report_eta(self.steps, self.total, epoch)
            epoch += 1
            state = dict(epoch=epoch ,
                     iter=0,
                     state_dict=model.state_dict(),
                     optimizer=optimizer.state_dict())
            self.model_saver.save_train(state,self.logger)
            if self.experiment.validation and\
                    ((self.steps % self.experiment.validation.interval == 0 and\
                    self.steps >= self.experiment.validation.exempt) or self.val_best>0): 
                if self.save_before_val:
                    self.save_before_val =False
                    self.model_saver.save_val_before(state,self.logger)
                self.validate(validation_loaders, model, epoch, self.steps)
            
            
            
            if epoch > self.experiment.train.epochs:
                self.model_saver.save_checkpoint(model, 'final')
                if self.experiment.validation:
                    self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.info('Training done')
                break
            iter_delta = 0
        #self.writer.flush()

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        optimizer.zero_grad()
        results = model.forward(batch,  training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results
        # print(metrics)
        # print(l)
        self.losses.update(l.mean())
        for  v in metrics:
            # print(v)
            self.losses_item[v].update(metrics[v].mean())
            self.losses_item[v].update(metrics[v].mean())
        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()

        loss.backward()
        optimizer.step()

        if step % self.experiment.logger.log_interval == 0:
            # print(step)
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                    step, epoch, self.losses.avg, self.current_lr))
                # self.logger.info('99999')
            self.logger.add_scalar('loss/loss', self.losses.avg, epoch)
            for  v in metrics:
                self.logger.add_scalar('loss/'+v, self.losses_item[v].avg, epoch)
                self.logger.info('%s: %6f' % (v, self.losses_item[v].avg))
            self.logger.add_scalar('learning_rate', self.current_lr, epoch)
            self.logger.report_time('Logging')

    def validate(self, validation_loaders, model, epoch, step):
        all_matircs = {}
        model.eval()
        for name, loader in validation_loaders.items():
            if self.experiment.validation.visualize:
                metrics, vis_images = self.validate_step(
                    loader, model, True)
                self.logger.images(
                    os.path.join('vis', name), vis_images, step)
            else:
                metrics, vis_images = self.validate_step(loader, model.model.module, False)
            for _key, metric in metrics.items():
                key = name + '/' + _key
                if key in all_matircs:
                    all_matircs[key].update(metric.val, metric.count)
                else:
                    all_matircs[key] = metric

        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        if all_matircs['icdar2015/fmeasure'].avg>self.f1:
            self.f1 = all_matircs['icdar2015/fmeasure'].avg
            self.best_epoch = epoch
            self.best_pre = all_matircs['icdar2015/precision'].avg
            self.best_recall = all_matircs['icdar2015/recall'].avg
            self.model_saver.best_save_model(
                model, epoch ,self.steps,round(self.f1,5),self.logger)
            if self.args.get('finetune_iter',False):
                self.val_best =3
        self.logger.info('best F1 : %f best epoch:%d best pre: %f best recall :%f' %(
                self.f1,self.best_epoch,self.best_pre,self.best_recall))
        self.logger.add_scalar('F-score', all_matircs['icdar2015/fmeasure'].avg, epoch)   

        model.train()
        return all_matircs
    def copy_cur_env(self, work_dir, dst_dir):
       
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        for filename in os.listdir(work_dir):

            file = os.path.join(work_dir,filename)
            dst_file = os.path.join(dst_dir,filename)

            if os.path.isdir(file) and 'workspace' not in filename and 'output' not in filename\
                and 'model' not in filename and 'results' not in filename and 'runs' not in filename:
                shutil.copytree(file, dst_file)
            elif os.path.isfile(file):
                shutil.copyfile(file,dst_file)
    def validate_step(self, data_loader, model, visualize=False):
        raw_metrics = []
        vis_images = dict()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            with torch.no_grad():
                
                image = batch['image']
                image = Variable(image).cuda()
                pred = model.forward(image, training=False)
                # output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])
                # pred = pred.cpu().numpy()
                # pred = model.forward(batch, training=False,visualize = self.visualize)
                pred = pred.cpu().numpy()
                height, width = batch['shape'][0].cpu().numpy()
                output = self.structure.representer.represent(batch, pred, height, width, is_output_polygon=self.args['polygon'])  
                raw_metric = self.structure.measurer.validate_measure(
                    batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                raw_metrics.append(raw_metric)

                # if visualize and self.structure.visualizer:
                #     vis_image = self.structure.visualizer.visualize(
                #         batch, output, interested)
                #     vis_images.update(vis_image)
        metrics = self.structure.measurer.gather_measure(
            raw_metrics, self.logger)
        return metrics, vis_images

    def to_np(self, x):
        return x.cpu().data.numpy()
