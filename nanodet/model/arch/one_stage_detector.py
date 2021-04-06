import time
import torch
import torch.nn as nn
from nanodet.model.backbone import build_backbone
from nanodet.model.fpn import build_fpn
from nanodet.model.head import build_head


class OneStageDetector(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg=None,
                 head_cfg=None,):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if backbone_cfg.name == 'ShuffleNetV2' and fpn_cfg is not None:
            # adaptively change fpn_cfg.in_channels according backbone_cfg.model_size
            model_size = backbone_cfg.model_size
            if model_size == '0.5x':
                fpn_cfg.in_channels = [48, 96, 192]
            elif model_size == '1.0x':
                fpn_cfg.in_channels = [116, 232, 464]
            elif model_size == '1.5x':
                fpn_cfg.in_channels = [176, 352, 704]
            elif model_size == '2.0x':
                fpn_cfg.in_channels = [244, 488, 976]
            else:
                raise NotImplementedError
        self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'fpn'):
            x = self.fpn(x)
        if hasattr(self, 'head'):
            x = self.head(x)
        return x

    def inference(self, meta):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            results = self.head.post_process(preds, meta)
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta['img'])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states


if __name__ == "__main__":
    from nanodet.util import cfg, load_config

    args_config = r'../../../config/nanodet-m.yml'
    load_config(cfg, args_config)
    detector = OneStageDetector(cfg.model.arch.backbone, cfg.model.arch.fpn, cfg.model.arch.head)
    print(detector)