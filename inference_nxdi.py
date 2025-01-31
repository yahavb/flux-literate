# inference_nxdi.py
import argparse
import torch
import torch.nn as nn
import torch_neuronx
import neuronx_distributed
import os
import math
import time
import numpy as npy
from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Any, Dict, Optional, Union, List, Tuple
from transformers.models.t5.modeling_t5 import T5EncoderModel
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
import logging
from diffusers.models.transformers import FluxTransformer2DModel
from types import SimpleNamespace
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxTransformer2DModel as NeuronFluxTransformer2DModelNew
from neuronx_distributed.trace.spmd import NxDModelExecutor
from neuronx_distributed_inference.modules.checkpoint import load_state_dict
from neuronx_distributed_inference.models.diffusers.padder import MaybePadder
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


from torch.profiler import profile, record_function, ProfilerActivity
from safetensors.torch import load_file


logger = logging.getLogger("Test")
logger.setLevel(logging.INFO)

CKPT_DIR = "/home/ubuntu/models/FLUX.1-dev/transformer/"
BASE_COMPILE_WORK_DIR = "/home/ubuntu/workplace/DiffusersTest/src/KaenaDiffusersTest/test_src/flux/trn1/nxdi_backbone/"

if not os.path.exists(BASE_COMPILE_WORK_DIR):
    os.makedirs(BASE_COMPILE_WORK_DIR)

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)

TEXT_ENCODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_1/compiled_model/model.pt')

TEXT_ENCODER_2_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_2/compiled_model')

VAE_DECODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'decoder/compiled_model/model.pt')

EMBEDDERS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/embedders')
OUT_LAYERS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/out_layers')
SINGLE_TRANSFORMER_BLOCKS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/single_transformer_blocks')
TRANSFORMER_BLOCKS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/transformer_blocks')


def get_compiler_args(dtype=torch.float32):
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1 "
    # Flag for model type
    compiler_args += "--model-type=transformer"
    # Add flags for cc-overlaptrace
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    # Prevent auto-down casting when running with fp32
    if dtype == torch.float32:
        compiler_args += " --auto-cast=none"
    print(f"compiler_args: {compiler_args}")
    return compiler_args


def get_hf_checkpoint_loader_fn():
    logger.info(f"Loading real checkpoint from {CKPT_DIR}")
    load1 = load_file(os.path.join(CKPT_DIR, "diffusion_pytorch_model-00001-of-00003.safetensors"))
    load2 = load_file(os.path.join(CKPT_DIR, "diffusion_pytorch_model-00002-of-00003.safetensors"))
    load3 = load_file(os.path.join(CKPT_DIR, "diffusion_pytorch_model-00003-of-00003.safetensors"))
    merged_state_dict = {**load1, **load2, **load3}
    return merged_state_dict


def get_checkpoint_loader_fn():
    state_dict = state_dict = get_hf_checkpoint_loader_fn()
    inner_dim = model_config["num_attention_heads"] * model_config["attention_head_dim"]
    for i in range(model_config["num_single_layers"]):
        # the shape of weights is [out_size, in_size], we want to split the in_size dimension
        # there is no need to split the bias, because is shape is [out_size]
        state_dict[f"single_transformer_blocks.{i}.proj_out_attn.weight"] = state_dict[
            f"single_transformer_blocks.{i}.proj_out.weight"
        ][:, : inner_dim].contiguous()
        state_dict[f"single_transformer_blocks.{i}.proj_out_attn.bias"] = (
            state_dict[f"single_transformer_blocks.{i}.proj_out.bias"]
            .clone()
            .detach()
            .contiguous()
        )
        state_dict[f"single_transformer_blocks.{i}.proj_out_mlp.weight"] = state_dict[
            f"single_transformer_blocks.{i}.proj_out.weight"
        ][:, inner_dim :].contiguous()
    return state_dict

def trace_nxd_model(example_inputs, model_cls):
    model_builder = ModelBuilder(
        router=None,
        debug=debug,
        tp_degree=tp_degree,
        checkpoint_loader=get_checkpoint_loader_fn,
        compiler_workdir=BASE_COMPILE_WORK_DIR,
    )
    logger.info("Initiated model builder!")

    def create_model():
        model = model_cls(**model_config)
        model.eval()
        if dtype == torch.bfloat16:
            model.bfloat16()
        return model

    model_builder.add(
        key="test_flux_model",
        model_instance=BaseModelInstance(
            module_cls=create_model, input_output_aliases={}
        ),
        example_inputs=[example_inputs],
        priority_model_idx=0,
        compiler_args=get_compiler_args(dtype),
    )
    logger.info("Added model builder! Starting to trace!")
    start_time = time.time()

    traced_model = model_builder.trace(initialize_model_weights=True)
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))

    elapsed_time = time.time() - start_time
    logger.info(f"Traced time taken {elapsed_time} s")

    logger.info("Done tracing the model!")
    return traced_model


class NeuronFluxCLIPTextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        output = self.encoder(emb)
        output = CLIPEncoderOutput(output)
        return output


class CLIPEncoderOutput():
    def __init__(self, dictionary):
        self.pooler_output = dictionary["pooler_output"]


class InferenceTextEncoderWrapper(nn.Module):
    def __init__(self, dtype, t: T5EncoderModel, seqlen: int):
        super().__init__()
        self.dtype = dtype
        self.device = t.device
        #self.t = t
        self.t = neuronx_distributed.trace.parallel_model_load(TEXT_ENCODER_2_PATH)
    def forward(self, text_input_ids, output_hidden_states=False):
        return [self.t(text_input_ids)['last_hidden_state'].to(self.dtype)]


class NeuronFluxTransformer2DModelV2(nn.Module):
    def __init__(self, dtype, transformer):
        super().__init__()
        self.dtype = dtype
        self.transformer = transformer
        self.device = torch.device("cpu")

        self.config = InferenceConfig(
            neuron_config=NeuronConfig(
                batch_size=1,
                torch_dtype=torch.bfloat16,
            ),
            patch_size=1,
            in_channels=64,
            out_channels=None,
            num_layers=19,
            num_single_layers=38,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            axes_dims_rope=(16, 56, 56),
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
    ):
        inputs = [
        hidden_states.bfloat16(),
        encoder_hidden_states.bfloat16(),
        pooled_projections.bfloat16(),
        timestep.bfloat16(),
        img_ids.bfloat16(),
        txt_ids.bfloat16(),
        guidance.bfloat16(),
        ]

    
        # Call the transformer with the prepared inputs
        output = self.transformer(*inputs)

        # Convert output back to bfloat16 if necessary
        if self.dtype == torch.bfloat16:
            if isinstance(output, torch.Tensor):
                output = output.bfloat16()
            elif isinstance(output, dict):
                output = {k: v.bfloat16() if isinstance(v, torch.Tensor) else v for k, v in output.items()}

        # Return the output directly
        return output



def get_neuron_backbone():
    test_inputs = (
            torch.randn([1, 4096, 64], dtype=dtype),
            torch.randn([1, 512, 4096], dtype=dtype),
            torch.randn([1, 768], dtype=dtype),
            torch.randn([1], dtype=dtype),
            torch.randn([4096, 3], dtype=dtype),
            torch.randn([512, 3], dtype=dtype),
            torch.tensor([3.5], dtype=dtype),  # guidance
        )
    
    neuron_model = trace_nxd_model(test_inputs, NeuronFluxTransformer2DModelNew)
    return neuron_model

def run_inference(
        prompt,
        height,
        width,
        max_sequence_length,
        num_inference_steps,
        instance_type):
    if instance_type == "trn2":
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        os.environ["LOCAL_WORLD_SIZE"] = "4"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        # cache_dir="flux_hf_cache_dir",
        )

    pipe.text_encoder = NeuronFluxCLIPTextEncoderModel(
        pipe.text_encoder.dtype,
        torch.jit.load(TEXT_ENCODER_PATH))
    
    text_encoder_2_wrapper = InferenceTextEncoderWrapper(torch.bfloat16, pipe.text_encoder_2, max_sequence_length)
    # traced model loaded within InferenceTextEncoderWrapper init
    pipe.text_encoder_2 = text_encoder_2_wrapper

    neuron_backbone = get_neuron_backbone()

    pipe.transformer = NeuronFluxTransformer2DModelV2(torch.bfloat16, neuron_backbone)
    # print("the neuron model config", neuron_model.config)
    
    pipe.vae.decoder = torch.jit.load(VAE_DECODER_PATH)

    # First do 5 warmup runs so all the asynchronous loads can finish and times can stabilize
    for i in range(5):
        image_warmup = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length
        ).images[0]

    image_warmup.save(f"image.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="prompt for image to be generated; generates cat by default"
    )
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
        "-n",
        "--num_inference_steps",
        type=int,
        default=25,
        help="number of inference steps to run in generating image"
    )
    parser.add_argument(
        "-i",
        "--instance_type",
        type=str,
        default="trn1",
        help="instance type to run this model on, e.g. trn1, trn2"
    )
    args = parser.parse_args()
    model_config = {
        "attention_head_dim": 128,
        "guidance_embeds": True,
        "in_channels": 64,
        "joint_attention_dim": 4096,
        "num_attention_heads": 24,
        "num_layers": 19,
        "num_single_layers": 38,
        "patch_size": 1,
        "pooled_projection_dim": 768,
    }
    dtype = torch.bfloat16
    tp_degree = 8
    debug = False
    run_inference(
        args.prompt,
        args.height,
        args.width,
        args.max_sequence_length,
        args.num_inference_steps,
        args.instance_type)
