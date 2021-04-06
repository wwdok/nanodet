"""
该模块里的函数每一个特征图level执行一次，一个iteration里有3个level，所以处理的张量shape基本都是1个batch size所形成的shape
本用例用的batch size=8，类别数=9
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .utils import weighted_loss
# logging.basicConfig(filename='logging_gfocal_loss.txt', level=logging.INFO)


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N)
            and target quality label with shape (N). N=batch_size * featuremap_size
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    # logging.info('---------------------quality_focal_loss--------------------------')
    # logging.info(f'label is {label} and size is {label.size()}')  # size分别有[12800],[3200],[800]，12800=8×40×40，8是batch size，40×40是feature map size，3200和800以此类推
    # 因为[3200][800]的输出内容较多会截断，所以要看内部数值就看[800]的输出,大部分数值是9，9代表的是背景，0到8是我们定义的具体类别id
    # logging.info(f'score is {score} and size is {score.size()}')  # 同上，大部分是0，少部分是0到1之间的小数，具体情况看[800]时的输出

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    # logging.info(f'scale_factor is {scale_factor} and size is {scale_factor.size()}')  # [12800, 9], [3200, 9], [800, 9]个0到1之间的小数
    zerolabel = scale_factor.new_zeros(pred.shape)
    # logging.info(f'zerolabel is {zerolabel} and size is {zerolabel.size()}')  # [12800, 9], [3200, 9], [800, 9]个0
    # logging.info(f'pred is {pred} and size is {pred.size()}')  # [12800, 9], [3200, 9], [800, 9]个正数和负数
    # GFL分成两部分计算，先计算负样本的损失，计算量是[12800, 9], [3200, 9], [800, 9]个交叉熵
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    # logging.info(f'loss is {loss} and size is {loss.size()}')  # [12800, 9], [3200, 9], [800, 9]
    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    # logging.info(f'bg_class_ind is {bg_class_ind}')  # 9，即类别数
    pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(1)
    # logging.info(f'pos is {pos} and size is {pos.size()}')  # pos.size()因每个batch不同而不同，里面的数值都是非背景类的list index
    pos_label = label[pos].long()
    # logging.info(f'pos_label is {pos_label} and size is {pos_label.size()}')  # pos_label.size()跟pos.size()一样，里面的数值是0到8之间的类别id
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    # logging.info(f'scale_factor is {scale_factor} and size is {scale_factor.size()}')
    # GFL分成两部分计算，再计算正样本的损失，但是这次只更新loss里pos.size()个正样本所在位置的损失值
    # loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
    #     pred[pos, pos_label], score[pos],
    #     reduction='none') * scale_factor.abs().pow(beta)
    # 我将代码改成特定类别（其数据量较少）的权重放大3倍，讨论贴：https://discuss.pytorch.org/t/question-about-the-usage-of-tensors-used-as-indices/116950
    updated_value = F.binary_cross_entropy_with_logits(pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)
    boolean_mask = sum(pos_label.eq(i) for i in [1, 2, 3]).bool()
    updated_value[boolean_mask] = 3 * updated_value[boolean_mask]
    loss[pos, pos_label] = updated_value
    # logging.info(f'loss is {loss} and size is {loss.size()}')  # [12800, 9], [3200, 9], [800, 9]
    # logging.info('===========================quality_focal_loss===========================')
    # GFL分成两部分计算，相加正样本和负样本的损失
    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    # logging.info('---------------------distribution_focal_loss--------------------------')
    # logging.info(f'label is {label} and size is {label.size()}')  # label.size() = 4 * pos.size()
    dis_left = label.long()  # torch.long = torch.int64, 这里long()会把小数向下取整，有点类似floor(),不过floor完还是浮点类型，而long()完还变成了整数类型
    # logging.info(f'dis_left is {dis_left} and size is {dis_left.size()}')
    dis_right = dis_left + 1
    # logging.info(f'dis_right is {dis_right} and size is {dis_right.size()}')
    # logging.info(f'dis_right.float() is {dis_right.float()}')
    weight_left = dis_right.float() - label
    # logging.info(f'weight_left is {weight_left} and size is {weight_left.size()}')
    # logging.info(f'dis_left.float() is {dis_left.float()}')
    weight_right = label - dis_left.float()
    # logging.info(f'weight_right is {weight_right} and size is {weight_right.size()}')
    # logging.info(f'pred is {pred} and size is {pred.size()}')
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    # logging.info('===========================distribution_focal_loss===========================')
    return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls
