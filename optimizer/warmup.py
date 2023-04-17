from torch import optim

class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler, warmup_batch):
        self.scheduler = scheduler
        self.warmup_batch = warmup_batch
        self.scheduler 
        super(WarmupLR, self).__init__(optimizer)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_batch:
            warmup_factor = self.last_epoch / self.warmup_batch
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_batch:
            if epoch is None:
                self.scheduler.step(None)
            else:
                self.scheduler.step(epoch - self.warmup_batch)
        else:
            return super(WarmupLR, self).step(epoch)