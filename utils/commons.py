import torch
import math
import os
from torch import nn

from neuronxcc.nki.language import nc as vnc
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)


def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape
        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])
    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)
    attn_out = torch.bmm(attention_probs, value)
    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )
    return attn_out



def attention_wrapper_sharded_without_swap(query, key, value):
                 
        bs, n_head, q_len, d_head = query.shape  #my change    
        q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
        k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
        v = value.clone().reshape((bs*n_head, q_len, d_head))
                
        attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

        
        vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
        use_sharded_attention_kernel = (vc_size == 2)
        scale = 1 / math.sqrt(d_head)
        
        if use_sharded_attention_kernel:
            grid = (vnc(2),)
            _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
        else:
            _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
        
        attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
              
        return attn_output


sdpa_original = torch.nn.functional.scaled_dot_product_attention
def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    else:
        return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
       
        
def attention_wrapper_for_transformer(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    else:
        return attention_wrapper_sharded_without_swap(query, key, value)
