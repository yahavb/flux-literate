import argparse
import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from model import TracingCLIPTextEncoderWrapper

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)


def trace_text_encoder(instance_type):
    compiler_args = "--enable-fast-loading-neuron-binaries"
    if instance_type == "trn2":
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        compiler_args = "--target=trn2 --lnc=2 --enable-fast-loading-neuron-binaries"
    pipe = FluxPipeline.from_pretrained(
        "optimum-internal-testing/tiny-random-flux",
        torch_dtype=torch.bfloat16,
        cache_dir="flux_hf_cache_dir")
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe

    text_encoder = TracingCLIPTextEncoderWrapper(text_encoder)

    emb = torch.zeros((1, 77), dtype=torch.int64)

    text_encoder_neuron = torch_neuronx.trace(
        text_encoder.neuron_text_encoder,
        emb,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=compiler_args,
        inline_weights_to_neff=False,
        )

    torch_neuronx.async_load(text_encoder_neuron)

    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                         'compiled_model/model.pt')
    torch.jit.save(text_encoder_neuron, text_encoder_filename)

    del text_encoder
    del text_encoder_neuron


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--instance_type",
        type=str,
        default="trn2",
        help="instance type to run this model on, e.g. trn1, trn2"
    )
    args = parser.parse_args()
    trace_text_encoder(args.instance_type)

