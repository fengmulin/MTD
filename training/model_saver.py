import os

import torch

from concern.config import Configurable, State
from concern.signal_monitor import SignalMonitor

class ModelSaver(Configurable):
    dir_path = State()
    save_interval = State(default=1000)
    signal_path = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        self.monitor = SignalMonitor(self.signal_path)

    def maybe_save_model(self, model, epoch,  logger):
        if epoch % self.save_interval == 0 or self.monitor.get_signal() is not None :
            self.save_model(model, epoch)
            logger.report_time('Saving ')
            #logger.iter(step)
            
    def best_save_model(self, model, epoch, steps, f1,logger):
        #if step % self.save_interval == 0 or self.monitor.get_signal() is not None and step>1:
        self.save_model(model, epoch,steps,f1)
        logger.report_time('Saving ')
        #logger.iter(step)
    def save_train(self, state,logger):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        torch.save(state, os.path.join(self.dir_path, 'checkpoint.pth.tar'))
        logger.report_time('Saving ')
    def save_val_before(self, state,logger):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        torch.save(state, os.path.join(self.dir_path, 'val_before_checkpoint.pth.tar'))
        logger.report_time('Saving ')
    def save_model(self, model, epoch=None,  step=None,f1=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = self.make_checkpoint_name(name, epoch,step, f1)
                self.save_checkpoint(net, checkpoint_name)
        else:
            checkpoint_name = self.make_checkpoint_name('model', epoch,step,  f1)
            self.save_checkpoint(model, checkpoint_name)

    def save_checkpoint(self, net, name):
        os.makedirs(self.dir_path, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(self.dir_path, name))

    def make_checkpoint_name(self, name, epoch=None, step=None, f1=None):
        if epoch is None :
            c_name = name + '_latest'
        elif  step is None:
            c_name = '{}_epoch_{}_f1_{}'.format(name, epoch, f1)
        else:
            c_name = '{}_epoch_{}_step_{}_f1_{}'.format(name, epoch, step, f1)
        return c_name
