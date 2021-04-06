import os
import torch
import logging
import argparse
import numpy as np
import torch.distributed as dist
import time
from nanodet.util import mkdir, Logger, cfg, load_config
from nanodet.trainer import build_trainer
from nanodet.data.collate import collate_function
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.evaluator import build_evaluator

"""
不要在pycharm里一键运行该脚本启动训练。不然生成的workspace子文件夹和wandb子文件夹都会被保存在tools里，而且config yml和demo.py的路径也要跟着改
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/EfficientNet-Lite/nanodet-EfficientNet-Lite0_320_special_vehicle_9classes.yml', help='train config file path')  # config/nanodet_m_416_special_vehicle_9classes.yml
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='whether to use wandb to record the training process')
    args, unknown = parser.parse_known_args()
    return args, unknown


def init_seeds(seed=0):
    """
    manually set a random seed for numpy, torch and cuda
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args, unknown):
    load_config(cfg, args.config)
    """
    Following block of code is for integrated with wandb sweep, after you run sth like `wandb agent wadewang/NanoDet_SpecialVehicle/82y9e619`
    wandb will generate command like `` to execute, original argparse can't not handle these new added arguments. Then i 
    use https://stackoverflow.com/a/48057478/12169382 and https://github.com/rbgirshick/yacs#command-line-overrides to solve this problem.
    """
    # print(f'Previous cfg is {cfg}')
    opt_list = list()
    if len(unknown) != 0:
        for c in range(len(unknown)):
            # print(unknown[c])  # eg:'--data.train.pipeline.brightness=0.462972446342085'
            # print(unknown[c].split('--'))  # e.g:['', 'data.train.pipeline.brightness=0.462972446342085']
            opt_list.extend(unknown[c].split('--')[1].split('='))
    # print(opt_list)
    cfg.merge_from_list(opt_list)
    # print(f'Present cfg is {cfg}')
    # assert False  # add stop point here to facilitate debugging

    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # add save_dir_extension to avoid overrider previous folder when sweep
    save_dir_extension = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    cfg.save_dir = cfg.save_dir + save_dir_extension
    opt_list.extend(['save_dir',cfg.save_dir])
    mkdir(local_rank, cfg.save_dir)
    logger = Logger(local_rank, cfg.save_dir)
    wandb = None
    if args.use_wandb:
        try:
            import wandb
            wandb.login()
        except ImportError:
            wandb = None
            logger.log(f"wandb: To use Weights & Biases for NanoDet logging, please install it with 'pip install wandb' first")
    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        init_seeds(args.seed)

    logger.log('Creating model...')
    model = build_model(cfg.model)

    logger.log('Setting up data...')
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')

    if len(cfg.device.gpu_ids) > 1:
        print('rank = ', local_rank)
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       num_workers=cfg.device.workers_per_gpu, pin_memory=True,
                                                       collate_fn=collate_function, sampler=train_sampler,
                                                       drop_last=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       shuffle=True, num_workers=cfg.device.workers_per_gpu,
                                                       pin_memory=True, collate_fn=collate_function, drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                 pin_memory=True, collate_fn=collate_function, drop_last=True)

    trainer = build_trainer(local_rank, cfg, model, logger, wandb)

    if 'load_model' in cfg.schedule:
        trainer.load_model(cfg)
    if 'resume' in cfg.schedule:
        trainer.resume(cfg)

    evaluator = build_evaluator(cfg, val_dataset)

    logger.log('Starting training...')
    trainer.run(train_dataloader, val_dataloader, evaluator)
    trained_model = os.path.join(cfg.save_dir, "model_best/model_best.pth")
    print(f'opt_list is {opt_list}')
    opt_list_string = ' '.join(opt_list)  # convert list to string, or the argparse will make it as ["['device.batchsize_per_gpu',","'16',"......
    inderence_command = f'python demo/demo.py image --config {args.config} --cfg_list {opt_list_string} --model {trained_model} --path demo/input'
    print(inderence_command)
    os.system(inderence_command)
    # 要运行wandb.save()的话，需要先去右键设置保存文件夹的属性-安全-Users完全控制，这样本程序才有权限写入文件
    # wandb.save(trained_model)  # upload model ckpt to wandb cloud if necessary, there is 100 GB free storage
    wandb.run.finish() if wandb and wandb.run else None
    # To further optimize hyper-parameters with Sweep, please visit : https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args, unknown = parse_args()
    # print(f'unknown is {unknown}')
    main(args, unknown)
