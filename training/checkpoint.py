from concern.config import Configurable, State
import os
import torch


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()
    recover = State()
    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']
        if 'recover' in cmd:
            self.recover = cmd['recover']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return
        # print(self.resume)
        # raise
        if not os.path.exists(self.resume):
            logger.warning("Checkpoint not found: " + self.resume)
            return

        #logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)
    def recover_model(self, model, device, logger,optimizer):
        if self.recover is None:
            return

        if not os.path.exists(self.recover):
            logger.warning("Checkpoint not found: " + self.recover)
            return
        checkpoint = torch.load(self.recover)
        self.start_epoch = checkpoint['epoch']
        self.start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #logger.info("Resuming from " + self.resume)
        #state_dict = torch.load(checkpoint['state_dict'], map_location=device)
        #model.load_state_dict(state_dict, strict=True)
        logger.info("Recoverd from " + self.recover)
        
    def restore_counter(self):
        return self.start_epoch, self.start_iter
    
    
