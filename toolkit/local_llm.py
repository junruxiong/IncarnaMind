"""The below code is borrowed from: https://github.com/PromtEngineer/localGPT
The reason to use gguf/ggml models: https://huggingface.co/TheBloke/wizardLM-7B-GGML/discussions/3"""
import logging
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from langchain.llms import LlamaCpp, HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    pipeline,
)
from toolkit.utils import Config


configs = Config("configparser.ini")
logger = logging.getLogger(__name__)


def load_gguf_hf_model(
    model_id: str,
    model_basename: str,
    max_tokens: int,
    temperature: float,
    device_type: str,
):
    """
    Load a GGUF/GGML quantized model using LlamaCpp.

    This function attempts to load a GGUF/GGML quantized model using the LlamaCpp library.
    If the model is of type GGML, and newer version of LLAMA-CPP is used which does not support GGML,
    it logs a message indicating that LLAMA-CPP has dropped support for GGML.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - model_basename (str): The base name of the model file.
    - max_tokens (int): The maximum number of tokens to generate in the completion.
    - temperature (float): The temperature of LLM.
    - device_type (str): The type of device where the model will run, e.g., 'mps', 'cuda', etc.

    Returns:
    - LlamaCpp: An instance of the LlamaCpp model if successful, otherwise None.

    Notes:
    - The function uses the `hf_hub_download` function to download the model from the HuggingFace Hub.
    - The number of GPU layers is set based on the device type.
    """

    try:
        logger.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=configs.local_model_dir,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": configs.max_llm_context,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_batch": configs.n_batch,  # set this based on your GPU & CPU RAM
            "verbose": False,
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = configs.n_gpu_layers  # set this based on your GPU

        return LlamaCpp(**kwargs)
    except:
        if "ggml" in model_basename:
            logger.info(
                "If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead"
            )
        return None


def load_full_hf_model(model_id: str, model_basename: str, device_type: str):
    """
    Load a full model using either LlamaTokenizer or AutoModelForCausalLM.

    This function loads a full model based on the specified device type.
    If the device type is 'mps' or 'cpu', it uses LlamaTokenizer and LlamaForCausalLM.
    Otherwise, it uses AutoModelForCausalLM.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - model_basename (str): The base name of the model file.
    - device_type (str): The type of device where the model will run.

    Returns:
    - model (Union[LlamaForCausalLM, AutoModelForCausalLM]): The loaded model.
    - tokenizer (Union[LlamaTokenizer, AutoTokenizer]): The tokenizer associated with the model.

    Notes:
    - The function uses the `from_pretrained` method to load both the model and the tokenizer.
    - Additional settings are provided for NVIDIA GPUs, such as loading in 4-bit and setting the compute dtype.
    """
    if "meta-llama" in model_id.lower():
        login(token=configs.huggingface_token)

    if device_type.lower() in ["mps", "cpu"]:
        logger.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            cache_dir=configs.local_model_dir,
        )
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            cache_dir=configs.local_model_dir,
        )
    else:
        logger.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=configs.local_model_dir
        )
        logger.info("Tokenizer loaded")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=configs.local_model_dir,
            # trust_remote_code=True, # set these if you are using NVIDIA GPU
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    return model, tokenizer


def load_local_llm(
    model_id: str,
    model_basename: str,
    temperature: float,
    max_tokens: int,
    device_type: str,
):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logger.info(f"Loading Model: {model_id}, on: {device_type}")
    logger.info("This action can take a few minutes!")

    if model_basename.lower() != "none":
        if ".gguf" in model_basename.lower():
            llm = load_gguf_hf_model(
                model_id, model_basename, max_tokens, temperature, device_type
            )
            return llm

    model, tokenizer = load_full_hf_model(model_id, None, device_type)
    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_tokens,
        temperature=temperature,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("Local LLM Loaded")

    return local_llm
