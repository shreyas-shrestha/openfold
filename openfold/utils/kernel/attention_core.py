# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F

# Try to import the CUDA kernel
try:
    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
except ModuleNotFoundError:
    attn_core_inplace_cuda = None


SUPPORTED_DTYPES = [torch.float32, torch.bfloat16]


class AttentionCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias_1=None, bias_2=None):
        if(bias_1 is None and bias_2 is not None):
            raise ValueError("bias_1 must be specified before bias_2")
        if(q.dtype not in SUPPORTED_DTYPES):
            raise ValueError("Unsupported datatype")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # [*, H, Q, K]
        attention_logits = torch.matmul(
            q, k.transpose(-1, -2),
        )

        if(bias_1 is not None):
            attention_logits += bias_1
        if(bias_2 is not None):
            attention_logits += bias_2

        # Check if our CUDA kernel exists. If so, use it.
        if attn_core_inplace_cuda is not None and q.device.type == "cuda":
            attn_core_inplace_cuda.forward_(
                attention_logits,
                reduce(mul, attention_logits.shape[:-1]),
                attention_logits.shape[-1],
            )
            # The attention_logits are modified in-place by the kernel
            attn_weights = attention_logits
        else:
            # Otherwise, use the standard PyTorch softmax.
            attn_weights = F.softmax(attention_logits, dim=-1)

        o = torch.matmul(attn_weights, v)

        ctx.save_for_backward(q, k, v, attn_weights, bias_1, bias_2)

        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attn_weights, bias_1, bias_2 = ctx.saved_tensors
        grad_q = grad_k = grad_v = grad_bias_1 = grad_bias_2 = None

        # Check if our CUDA kernel exists. If so, use its backward pass.
        if attn_core_inplace_cuda is not None and q.device.type == "cuda":
            # The custom kernel needs some specific shapes and inputs
            # The attn_weights tensor is modified in-place to become the gradient
            attn_core_inplace_cuda.backward_(
                attn_weights,
                grad_output.contiguous(),
                v.contiguous(),
                reduce(mul, attn_weights.shape[:-1]),
                attn_weights.shape[-1],
                grad_output.shape[-1],
            )
            grad_attn_logits = attn_weights # The tensor is now the gradient
        else:
            # Otherwise, let PyTorch's autograd handle it
            # This is the standard backward pass for the attention mechanism
            grad_attn_weights = torch.matmul(grad_output, v.transpose(-1, -2))
            
            # Backward pass for softmax
            sum_grad = torch.sum(grad_attn_weights * attn_weights, dim=-1, keepdim=True)
            grad_attn_logits = attn_weights * (grad_attn_weights - sum_grad)


        # Gradients for q, k, v
        grad_q = torch.matmul(grad_attn_logits, k)
        grad_k = torch.matmul(grad_attn_logits.transpose(-1, -2), q).transpose(-1, -2)
        grad_v = torch.matmul(attn_weights.transpose(-1, -2), grad_output)
        
        # Gradients for biases
        if bias_1 is not None:
            grad_bias_1 = torch.sum(
                grad_attn_logits,
                dim=tuple(i for i, d in enumerate(bias_1.shape) if d == 1),
                keepdim=True,
            )

        if bias_2 is not None:
            grad_bias_2 = torch.sum(
                grad_attn_logits,
                dim=tuple(i for i, d in enumerate(bias_2.shape) if d == 1),
                keepdim=True,
            )

        return grad_q, grad_k, grad_v, grad_bias_1, grad_bias_2

attention_core = AttentionCoreFunction.apply
