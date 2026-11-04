"""Check the W&B code path used by train_adaptive_model_hrnet.py without a GPU or network (offline mode).
Run on the cluster login node: WANDB_MODE=offline python tools/wandb_smoke.py"""
import os, tempfile, uuid
import wandb
d = tempfile.mkdtemp()
r = wandb.init(project=os.environ.get('WANDB_PROJECT', 'fidip'), name='smoke', id=uuid.uuid4().hex[:8],
               resume='allow', sync_tensorboard=True, dir=d, mode=os.environ.get('WANDB_MODE', 'offline'))
from torch.utils.tensorboard import SummaryWriter
w = SummaryWriter(log_dir=d); w.add_scalar('train_loss_P', 0.1, 0); w.close()
r.log({'val/AP': 0.5, 'epoch': 1}); r.summary['best_AP'] = 0.5; r.finish()
print('wandb smoke OK', wandb.__version__)
