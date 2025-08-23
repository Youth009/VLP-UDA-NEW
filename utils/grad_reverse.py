# utils/grad_reverse.py
import torch

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"GRL backward: lambda_={ctx.lambda_}, grad_output_mean={grad_output.mean().item()}")  # 添加调试信息
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)
