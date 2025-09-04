# cs336_basics/training.py
import math
import torch
import numpy.typing as npt
from collections.abc import Callable, Iterable


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            # 一些优化器（如 LBFGS）会多次调用 closure；AdamW 通常不需要，但保留接口
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # ===== MOD 1: 用 state['step'] 计数，并初始化状态 =====
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["m"]      # 一阶矩 m_t
                exp_avg_sq = state["v"]   # 二阶矩 v_t

                state["step"] += 1
                step = state["step"]

                # 更新一阶/二阶矩
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏置校正
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # ===== MOD 2: 用 no_grad + 原地 add_/addcdiv_；先权重衰减，再 Adam 更新 =====
                with torch.no_grad():
                    # decoupled weight decay: θ ← θ - lr * wd * θ
                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)

                    # Adam 参数更新: θ ← θ - step_size * m_hat / (sqrt(v_hat) + eps)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    # 提取目标对应的 logits 值
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))

    # 减去最大元素的 log-sum-exp 保证数值稳定性
    logsumexp = torch.logsumexp(inputs, -1, keepdim=True)

    # 计算损失时抵消 softmax 后的 log 和 exp 运算
    loss_matrix = -target_logits + logsumexp

    # 平均损失
    loss = torch.mean(loss_matrix)
    return loss


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:

    dataset_length = len(dataset)
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long)

    for i in range(batch_size):
        start_index = torch.randint(0, dataset_length - context_length, (1,)).item()
        input_seq = dataset[start_index : start_index + context_length]
        target_seq = dataset[start_index + 1 : start_index + context_length + 1]
        inputs[i] = torch.tensor(input_seq, dtype=torch.long)
        targets[i] = torch.tensor(target_seq, dtype=torch.long)

    if device:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
    return inputs, targets


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Cosine with warmup learning rate scheduler."""
    # 小于warmup_iters steps, 线性增长
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # 大于 cosine_cycle_iters, 最小值
    if it > cosine_cycle_iters:
        return min_learning_rate
    # 中间值, 余弦衰减
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def clip_gradient(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = 0.0

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef


def save_checkpoint(model, optimizer, iteration, out):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer):
    
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration