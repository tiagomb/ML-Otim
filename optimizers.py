import torch
from torch.optim.optimizer import Optimizer
import math

#Implementa uma versão simplificada do gradiente stocástico
class CustomSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError(f"Taxa de aprendizado inválida {lr}")
        
        defaults = dict(lr=lr)
        
        super(CustomSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                update_vec = p.grad
                
                #theta_t = theta_{t-1} - alpha * v_t
                p.add_(update_vec, alpha=-group['lr'])
                
        return loss

#Implementa versão simplificada do Adam 
class CustomAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Taxa de aprendizado inválida: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta 1 inválido: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta 2 inválido: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Média móvel exponencial dos gradientes (m)
                    state['exp_avg'] = torch.zeros_like(p)
                    # Média móvel exponencial dos gradientes ao quadrado (v)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                #m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                #v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                correction1 = 1 - beta1 ** t
                correction2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(correction2)).add_(eps)
                step_size = lr / correction1
                
                #p.data = p.data - step_size * (exp_avg / den)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
