from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4

config.TRAIN.beta1 = 0.9
config.TRAIN.beta1 = 0.999

## initialize G
config.TRAIN.n_epoch_init = 5000

config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'path for HR images'
config.TRAIN.lr_img_path = 'path for LR images'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'path for valid HR'
config.VALID.lr_img_path = 'path for valid LR'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
