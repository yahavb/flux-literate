import argparse
import copy
import neuronx_distributed
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux \
    import FluxTransformer2DModel
from model import (TracingTransformerEmbedderWrapper,
                   TracingTransformerBlockWrapper,
                   TracingSingleTransformerBlockWrapper,
                   TracingTransformerOutLayerWrapper,
                   init_transformer)
from utils.commons import attention_wrapper_for_transformer

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
torch.nn.functional.scaled_dot_product_attention = attention_wrapper_for_transformer


def trace_transformer_embedders():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="flux_hf_cache_dir")
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingTransformerEmbedderWrapper(
        transformer.x_embedder, transformer.context_embedder,
        transformer.time_text_embed, transformer.pos_embed)
    return mod_pipe_transformer_f, {}


def trace_transformer_blocks():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="flux_hf_cache_dir")
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingTransformerBlockWrapper(
        transformer, transformer.transformer_blocks)
    return mod_pipe_transformer_f, {}


def trace_single_transformer_blocks():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="flux_hf_cache_dir")
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingSingleTransformerBlockWrapper(
        transformer, transformer.single_transformer_blocks)
    return mod_pipe_transformer_f, {}


def trace_transformer_out_layers():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="flux_hf_cache_dir")
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingTransformerOutLayerWrapper(
        transformer.norm_out, transformer.proj_out)
    return mod_pipe_transformer_f, {}


def trace_transformer(height, width, max_sequence_length, instance_type):
    compiler_args = """--model-type=unet-inference"""
    tp_degree = 8
    if instance_type == "trn2":
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        compiler_args = """--target=trn2 --model-type=transformer --lnc=2 --enable-fast-loading-neuron-binaries -O1"""
        tp_degree = 4
        os.environ["LOCAL_WORLD_SIZE"] = "4"
    hidden_states = torch.rand([1, height * width // 256, 3072],
                               dtype=torch.bfloat16)
    timestep = torch.rand([1], dtype=torch.bfloat16)
    guidance = torch.rand([1], dtype=torch.float32)
    pooled_projections = torch.rand([1, 768], dtype=torch.bfloat16)
    sample_inputs = hidden_states, timestep, guidance, pooled_projections

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_transformer_embedders,
        sample_inputs,
        tp_degree=tp_degree,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=compiler_args,
        inline_weights_to_neff=False,
    )

    torch_neuronx.async_load(model)

    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/embedders')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model

    hidden_states = torch.rand([1, height * width // 256, 3072],
                               dtype=torch.bfloat16)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],
                                       dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    image_rotary_emb = (torch.rand(
        [height * width // 256 + max_sequence_length, 128],
        dtype=torch.bfloat16), torch.rand(
        [height * width // 256 + max_sequence_length, 128],
        dtype=torch.bfloat16))
    sample_inputs = hidden_states, encoder_hidden_states, \
        temb, image_rotary_emb
    
    if isinstance(image_rotary_emb, tuple):
        image_rotary_emb = torch.stack(image_rotary_emb, dim=-1)

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_transformer_blocks,
        sample_inputs,
        tp_degree=tp_degree,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )

    torch_neuronx.async_load(model)

    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/transformer_blocks')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model

    hidden_states = torch.rand(
        [1, height * width // 256 + max_sequence_length, 3072],
        dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    image_rotary_emb = (torch.rand(
        [height * width // 256 + max_sequence_length, 128],
        dtype=torch.bfloat16), torch.rand(
        [height * width // 256 + max_sequence_length, 128],
        dtype=torch.bfloat16))
    sample_inputs = hidden_states, temb, image_rotary_emb

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_single_transformer_blocks,
        sample_inputs,
        tp_degree=tp_degree,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )

    torch_neuronx.async_load(model)

    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/single_transformer_blocks')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model

    hidden_states = torch.rand(
        [1, height * width // 256 + max_sequence_length, 3072],
        dtype=torch.bfloat16)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],
                                       dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    sample_inputs = hidden_states, encoder_hidden_states, temb

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_transformer_out_layers,
        sample_inputs,
        tp_degree=tp_degree,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )

    torch_neuronx.async_load(model)

    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/out_layers')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hh",
        "--height",
        type=int,
        default=1024,
        help="height of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1024,
        help="width of images to be generated by compilation of this model"
    )
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
    trace_transformer(
        args.height,
        args.width,
        args.max_sequence_length,
        args.instance_type)

