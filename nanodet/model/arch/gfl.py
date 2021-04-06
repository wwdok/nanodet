from nanodet.model.arch.one_stage import OneStage


class GFL(OneStage):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg,
                 head_cfg, ):
        super(GFL, self).__init__(backbone_cfg,
                                  fpn_cfg,
                                  head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from nanodet.util import cfg, load_config

    args_config = r'../../../config/nanodet-m.yml'
    load_config(cfg, args_config)
    gfl = GFL(cfg.model.arch.backbone, cfg.model.arch.fpn, cfg.model.arch.head)
    print(gfl)
