from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence


def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    full_observations: Optional[Tensor] = None,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
    observed_ratio: Optional[Tensor] = None,
    observed_area_ratio: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 5 in
    [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).
    计算重构损失。当提供完整观测时，直接使用预测值与完整观测的差异作为损失。

    参数:
        po (Dict[str, Distribution]): 观测模型(decoder)返回的分布
        observations (Tensor): 环境提供的有限观测
        pr (Distribution): reward_model返回的奖励分布
        rewards (Tensor): 智能体在环境交互阶段获得的奖励
        priors_logits (Tensor): 先验的logits
        posteriors_logits (Tensor): 后验的logits
        full_observations (Tensor, optional): 环境的完整观测状态
        kl_dynamic (float): KL-balancing动态损失正则化器，默认为0.5
        kl_representation (float): KL-balancing表示损失正则化器，默认为0.1
        kl_free_nats (float): KL散度的下界，默认为1.0
        kl_regularizer (float): KL散度的缩放因子，默认为1.0
        pc (Distribution, optional): 终止步骤的预测分布，默认为None
        continue_targets (Tensor, optional): discount预测器的目标
        continue_scale_factor (float): continue损失的缩放因子，默认为1.0
        observed_ratio (Tensor, optional): 观测到的人群占总人群的比例
        observed_area_ratio (Tensor, optional): 无人机观测范围占整个场景的比例

    Returns:
        observation_loss (Tensor): 观测损失值
        KL divergence (Tensor): 后验和先验之间的KL散度
        reward_loss (Tensor): 奖励损失值
        state_loss (Tensor): 状态损失值
        continue_loss (Tensor): continue损失值（如果未计算则为0）
        reconstruction_loss (Tensor): 总重构损失值
    """
    device = rewards.device
    
    # 计算观测损失
    if full_observations is not None:
        # 如果有完整观测，直接使用完整观测计算损失
        observation_loss = -sum([po[k].log_prob(full_observations[k]) for k in po.keys()])
    else:
        # 如果没有完整观测，使用有限观测计算损失
        observation_loss = -sum([po[k].log_prob(observations[k]) for k in po.keys()])

    # 计算奖励损失
    reward_loss = -pr.log_prob(rewards)

    # KL balancing
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    
    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss

    # 计算continue损失（如果需要）
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)

    # 基于观测比例计算损失调整系数
    # 默认为1.0（不调整）
    loss_adjustment_factor = torch.tensor(1.0, device=device)
    
    if observed_ratio is not None and observed_area_ratio is not None:
        # 计算综合观测质量系数 (0.0-1.0范围)
        # 结合人群观测比例和区域观测比例
        # 权重可以调整，这里使用0.7和0.3的加权平均
        observation_quality = 0.7 * observed_ratio.mean() + 0.3 * observed_area_ratio.mean()
        
        # 创建损失削减系数：观测质量越高，削减越少
        # 公式: 1.0 - 削减率 * (1.0 - 观测质量)
        # 这里削减率设为0.5，可根据需要调整
        reduction_rate = 0.5
        loss_adjustment_factor = 1.0 - reduction_rate * (1.0 - observation_quality)
        
        # 确保系数在合理范围内
        loss_adjustment_factor = torch.clamp(loss_adjustment_factor, 0.5, 1.0)

    # 计算总重构损失并应用调整系数
    # 比例高时系数接近1，损失几乎不变
    # 比例低时系数接近0.5，损失减少约一半
    reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean() * loss_adjustment_factor
    
    return (
        reconstruction_loss,
        kl.mean(),
        kl_loss.mean(),
        reward_loss.mean(),
        observation_loss.mean(),
        continue_loss.mean(),
    )
