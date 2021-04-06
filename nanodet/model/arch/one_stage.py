import time
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class OneStage(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg=None,
                 head_cfg=None, ):
        super(OneStage, self).__init__()
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
        if hasattr(self, 'fpn') and self.fpn is not None:
            x = self.fpn(x)
        if hasattr(self, 'head'):
            out = []
            for xx in x:
                out.append(self.head(xx))
            x = tuple(out)
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
