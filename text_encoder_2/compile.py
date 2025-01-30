import argparse
import copy
import os
import torch
import math
import torch_neuronx
import neuronx_distributed
from diffusers import FluxPipeline
from torch import nn
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Block, T5LayerSelfAttention, T5LayerFF
from functools import partial

from utils.parallel_utils import get_sharded_data, shard_t5_self_attention, shard_t5_ff
from utils.commons import attention_wrapper

torch.nn.functional.scaled_dot_product_attention = attention_wrapper


COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)

class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)
    

class TracingT5WrapperTP(nn.Module):
    def __init__(self, t: T5EncoderModel, seqlen: int):
        super().__init__()
        self.t = t
        self.device = t.device
        precomputed_bias = self.t.encoder.block[0].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
        precomputed_bias_tp = get_sharded_data(precomputed_bias, 1)
        self.t.encoder.block[0].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias_tp
    
    # FLUX version
    def forward(self, text_input_ids):
        return self.t(
            text_input_ids, 
            output_hidden_states = False
        )

def get_text_encoder(tp_degree: int, sequence_length: int):
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        #"optimum-internal-testing/tiny-random-flux",
        torch_dtype=torch.bfloat16,
        cache_dir="flux_hf_cache_dir")
    
    text_encoder_2: T5EncoderModel = pipe.text_encoder_2
    text_encoder_2.eval()
    
    for idx, block in enumerate(text_encoder_2.encoder.block):
        block: T5Block = block
        block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        selfAttention: T5LayerSelfAttention = block.layer[0].SelfAttention
        ff: T5LayerFF = block.layer[1]
        layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
        layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)       
        block.layer[1] = shard_t5_ff(ff)
        block.layer[0].SelfAttention = shard_t5_self_attention(tp_degree, selfAttention)
        block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
        block.layer[1].layer_norm = f32Wrapper(layer_norm_1)
    final_layer_norm = pipe.text_encoder_2.encoder.final_layer_norm.to(torch.float32)
    pipe.text_encoder_2.encoder.final_layer_norm = f32Wrapper(final_layer_norm)             
    return TracingT5WrapperTP(text_encoder_2, sequence_length), {}


def trace_text_encoder_2(max_sequence_length, instance_type):
    
    compiler_flags = """--target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries"""
    if instance_type == "trn2":
        compiler_flags = """--target=trn2 --model-type=transformer --lnc=2 --enable-fast-loading-neuron-binaries -O1"""

    tp_degree = 8
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    if instance_type == "trn2":
        tp_degree = 4
        os.environ["LOCAL_WORLD_SIZE"] = "4"
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    get_text_encoder_f = partial(get_text_encoder, tp_degree, max_sequence_length)
    
    with torch.no_grad():     
        emb = torch.zeros((1, max_sequence_length), dtype=torch.int64)
        sample_inputs = emb
        
        compiled_text_encoder = neuronx_distributed.trace.parallel_model_trace(
            get_text_encoder_f,
            emb,
            compiler_workdir= os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )

    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                           'compiled_model')
    neuronx_distributed.trace.parallel_model_save(compiled_text_encoder, text_encoder_2_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=512,
        help="maximum sequence length for the text embeddings"
    )
    parser.add_argument(
        "-i",
        "--instance_type",
        type=str,
        default="trn2",
        help="instance type to run this model on, e.g. trn1, trn2"
    )
    args = parser.parse_args()
    trace_text_encoder_2(args.max_sequence_length, args.instance_type)

