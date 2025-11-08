import math
import os, torch, json, importlib
from typing import List

from .downloader import download_models, download_customized_models, Preset_model_id, Preset_model_website

from .sd_text_encoder import SDTextEncoder
from .sd_unet import SDUNet
from .sd_vae_encoder import SDVAEEncoder
from .sd_vae_decoder import SDVAEDecoder
from .lora import get_lora_loaders

from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sdxl_unet import SDXLUNet
from .sdxl_vae_decoder import SDXLVAEDecoder
from .sdxl_vae_encoder import SDXLVAEEncoder

from .sd3_text_encoder import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
from .sd3_dit import SD3DiT
from .sd3_vae_decoder import SD3VAEDecoder
from .sd3_vae_encoder import SD3VAEEncoder

from .sd_controlnet import SDControlNet
from .sdxl_controlnet import SDXLControlNetUnion

from .sd_motion import SDMotionModel
from .sdxl_motion import SDXLMotionModel

from .svd_image_encoder import SVDImageEncoder
from .svd_unet import SVDUNet
from .svd_vae_decoder import SVDVAEDecoder
from .svd_vae_encoder import SVDVAEEncoder

from .sd_ipadapter import SDIpAdapter, IpAdapterCLIPImageEmbedder
from .sdxl_ipadapter import SDXLIpAdapter, IpAdapterXLCLIPImageEmbedder

from .hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder, HunyuanDiTT5TextEncoder
from .hunyuan_dit import HunyuanDiT
from .hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
from .hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder
from .wan_video_dit import WanModel


def _format_shape(tensor):
    if tensor is None:
        return "missing"
    return "×".join(str(dim) for dim in tensor.shape)


def _extract_wan_config_from_state_dict(state_dict):
    config = {}
    patch_weight = state_dict.get("patch_embedding.weight")
    if patch_weight is not None:
        config["dim"] = int(patch_weight.shape[0])
        config["in_dim"] = int(patch_weight.shape[1])
        config["patch_size"] = tuple(int(d) for d in patch_weight.shape[2:])
    head_weight = state_dict.get("head.head.weight")
    if head_weight is not None and "patch_size" in config:
        patch_volume = 1
        for dim in config["patch_size"]:
            patch_volume *= dim
        if patch_volume:
            config["out_dim"] = int(head_weight.shape[0] // patch_volume)
    ffn_weight = state_dict.get("blocks.0.ffn.0.weight")
    if ffn_weight is not None:
        config["ffn_dim"] = int(ffn_weight.shape[0])
    time_weight = state_dict.get("time_embedding.0.weight")
    if time_weight is not None:
        config["freq_dim"] = int(time_weight.shape[1])
    text_weight = state_dict.get("text_embedding.0.weight")
    if text_weight is not None:
        config["text_dim"] = int(text_weight.shape[1])
    q_weight = state_dict.get("blocks.0.self_attn.q.weight")
    k_weight = state_dict.get("blocks.0.self_attn.k.weight")
    v_weight = state_dict.get("blocks.0.self_attn.v.weight")
    o_weight = state_dict.get("blocks.0.self_attn.o.weight")
    if q_weight is not None:
        config["self_attn_q_shape"] = tuple(int(d) for d in q_weight.shape)
    if k_weight is not None:
        config["self_attn_k_shape"] = tuple(int(d) for d in k_weight.shape)
    if v_weight is not None:
        config["self_attn_v_shape"] = tuple(int(d) for d in v_weight.shape)
    if o_weight is not None:
        config["self_attn_o_shape"] = tuple(int(d) for d in o_weight.shape)
    cross_q_weight = state_dict.get("blocks.0.cross_attn.q.weight")
    cross_k_weight = state_dict.get("blocks.0.cross_attn.k.weight")
    cross_v_weight = state_dict.get("blocks.0.cross_attn.v.weight")
    if cross_q_weight is not None:
        config["cross_attn_q_shape"] = tuple(int(d) for d in cross_q_weight.shape)
    if cross_k_weight is not None:
        config["cross_attn_k_shape"] = tuple(int(d) for d in cross_k_weight.shape)
    if cross_v_weight is not None:
        config["cross_attn_v_shape"] = tuple(int(d) for d in cross_v_weight.shape)
    block_indices = {
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("blocks.") and name.split(".")[1].isdigit()
    }
    if block_indices:
        config["num_layers"] = len(block_indices)
    return config


def _infer_wan_kwargs(state_dict, overrides=None):
    overrides = dict(overrides or {})
    inferred = {}

    config = _extract_wan_config_from_state_dict(state_dict)

    dim = config.get("dim")
    if dim is not None:
        inferred["dim"] = dim

    in_dim = config.get("in_dim")
    if in_dim is not None:
        inferred["in_dim"] = in_dim

    patch_size = config.get("patch_size")
    if patch_size is not None:
        inferred["patch_size"] = tuple(int(v) for v in patch_size)

    ffn_dim = config.get("ffn_dim")
    if ffn_dim is not None:
        inferred["ffn_dim"] = ffn_dim

    out_dim = config.get("out_dim")
    if out_dim is not None:
        inferred["out_dim"] = out_dim

    text_dim = config.get("text_dim")
    if text_dim is not None:
        inferred["text_dim"] = text_dim

    freq_dim = config.get("freq_dim")
    if freq_dim is not None:
        inferred["freq_dim"] = freq_dim

    num_layers = config.get("num_layers")
    if num_layers is not None:
        inferred["num_layers"] = num_layers

    q_shape = config.get("self_attn_q_shape")
    if dim is not None and q_shape is not None:
        _, q_in = q_shape
        head_dim = math.gcd(dim, q_in) if q_in else 0
        if head_dim:
            num_heads = dim // head_dim
            if num_heads == 0:
                num_heads = 1
            inferred["num_heads"] = num_heads

    if "num_heads" not in inferred and dim is not None and dim > 0:
        inferred["num_heads"] = max(1, dim // 128)

    inferred["eps"] = overrides.get("eps", 1e-6)

    has_image_input = any(name.startswith("img_emb") for name in state_dict)
    inferred["has_image_input"] = has_image_input if "has_image_input" not in overrides else overrides["has_image_input"]

    if any(name.startswith("img_emb.emb_pos") for name in state_dict):
        inferred["has_image_pos_emb"] = True

    if any(name.startswith("ref_conv") for name in state_dict):
        inferred["has_ref_conv"] = True

    if any(name.startswith("control_adapter") for name in state_dict):
        inferred["add_control_adapter"] = True
        conv_weight = state_dict.get("control_adapter.conv.weight")
        if conv_weight is not None and conv_weight.ndim >= 2:
            inferred["in_dim_control_adapter"] = max(1, int(conv_weight.shape[1] // 64))
    elif "add_control_adapter" not in overrides:
        inferred["add_control_adapter"] = False
        if "in_dim_control_adapter" not in overrides:
            inferred["in_dim_control_adapter"] = 24

    default_bools = {
        "seperated_timestep": False,
        "require_vae_embedding": True,
        "fuse_vae_embedding_in_latents": False,
    }
    if "require_clip_embedding" not in overrides:
        inferred["require_clip_embedding"] = True

    for key, value in default_bools.items():
        if key not in overrides:
            inferred[key] = value

    final_kwargs = dict(inferred)
    final_kwargs.update(overrides)
    return final_kwargs


def _raise_wan_shape_error(model, state_dict, exc):
    first_block = model.blocks[0] if model.blocks else None
    observed = {
        "q": _format_shape(state_dict.get("blocks.0.self_attn.q.weight")),
        "k": _format_shape(state_dict.get("blocks.0.self_attn.k.weight")),
        "v": _format_shape(state_dict.get("blocks.0.self_attn.v.weight")),
        "o": _format_shape(state_dict.get("blocks.0.self_attn.o.weight")),
    }
    expected = {}
    if first_block is not None:
        expected["q"] = _format_shape(first_block.self_attn.q.weight)
        expected["k"] = _format_shape(first_block.self_attn.k.weight)
        expected["v"] = _format_shape(first_block.self_attn.v.weight)
        expected["o"] = _format_shape(first_block.self_attn.o.weight)
    config_hint = _extract_wan_config_from_state_dict(state_dict)
    shape_rows = []
    for key in ("q", "k", "v", "o"):
        shape_rows.append(f"    {key}: checkpoint={observed.get(key)} | model={expected.get(key)}")
    config_lines = ["    " + name + "=" + str(value) for name, value in sorted(config_hint.items())]
    message = [
        "Failed to load WanModel weights from checkpoint due to incompatible attention shapes.",
        "Observed projection shapes:",
        *shape_rows,
    ]
    if config_lines:
        message.append("Inferred checkpoint configuration (for reference):")
        message.extend(config_lines)
        message.append(
            "If you are supplying manual model kwargs (e.g. WAN_GGUF_DIT_INIT), ensure they match the checkpoint's dimensions."
        )
    raise RuntimeError("\n".join(message)) from exc

from .flux_dit import FluxDiT
from .flux_text_encoder import FluxTextEncoder2
from .flux_vae import FluxVAEEncoder, FluxVAEDecoder
from .flux_ipadapter import FluxIpAdapter

from .cog_vae import CogVAEEncoder, CogVAEDecoder
from .cog_dit import CogDiT

from ..extensions.RIFE import IFNet
from ..extensions.ESRGAN import RRDBNet

from ..configs.model_config import model_loader_configs, huggingface_model_loader_configs, patch_model_loader_configs
from .utils import load_state_dict, init_weights_on_device, hash_state_dict_keys, split_state_dict_with_prefix


def load_model_from_single_file(
    state_dict,
    model_names,
    model_classes,
    model_resource,
    torch_dtype,
    device,
    manual_extra_kwargs=None,
):
    manual_extra_kwargs = manual_extra_kwargs or []
    loaded_model_names, loaded_models = [], []
    expanded_manual_kwargs = list(manual_extra_kwargs)
    if expanded_manual_kwargs and len(expanded_manual_kwargs) < len(model_names):
        last_kwargs = expanded_manual_kwargs[-1]
        expanded_manual_kwargs.extend([last_kwargs] * (len(model_names) - len(expanded_manual_kwargs)))
    elif not expanded_manual_kwargs:
        expanded_manual_kwargs = [{} for _ in model_names]

    for idx, (model_name, model_class) in enumerate(zip(model_names, model_classes)):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(f"        This model is initialized with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}
        manual_kwargs = expanded_manual_kwargs[idx] if idx < len(expanded_manual_kwargs) else {}
        combined_kwargs = {**extra_kwargs, **manual_kwargs}
        if issubclass(model_class, WanModel):
            combined_kwargs = _infer_wan_kwargs(model_state_dict, combined_kwargs)
        torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
        with init_weights_on_device():
            model = model_class(**combined_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()

        pretrained_keys = set(model_state_dict.keys())

 
        model_keys = set(model.state_dict().keys())


        missing_keys = model_keys - pretrained_keys
        
        if missing_keys:
            print(f"        The following parameters were not in the checkpoint and will be initialized: {sorted(list(missing_keys))}")

            for key in missing_keys:
     
                meta_param = model.get_parameter(key)
                

                print(f"            Initializing '{key}' with zeros. Shape: {meta_param.shape}, Dtype: {meta_param.dtype}")
                model_state_dict[key] = torch.zeros(
                    meta_param.shape, 
                    dtype=meta_param.dtype, 
                    device='cpu' 
                )
        try:
            model.load_state_dict(model_state_dict, assign=True,strict=False)
        except RuntimeError as exc:
            if isinstance(model, WanModel) and "size mismatch" in str(exc):
                _raise_wan_shape_error(model, model_state_dict, exc)
            raise
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_model_from_huggingface_folder(file_path, model_names, model_classes, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        if torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            model = model_class.from_pretrained(file_path, torch_dtype=torch_dtype).eval()
        else:
            model = model_class.from_pretrained(file_path).eval().to(dtype=torch_dtype)
        if torch_dtype == torch.float16 and hasattr(model, "half"):
            model = model.half()
        try:
            model = model.to(device=device)
        except:
            pass
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_single_patch_model_from_single_file(state_dict, model_name, model_class, base_model, extra_kwargs, torch_dtype, device):
    print(f"    model_name: {model_name} model_class: {model_class.__name__} extra_kwargs: {extra_kwargs}")
    base_state_dict = base_model.state_dict()
    base_model.to("cpu")
    del base_model
    model = model_class(**extra_kwargs)
    model.load_state_dict(base_state_dict, strict=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(dtype=torch_dtype, device=device)
    return model


def load_patch_model_from_single_file(state_dict, model_names, model_classes, extra_kwargs, model_manager, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        while True:
            for model_id in range(len(model_manager.model)):
                base_model_name = model_manager.model_name[model_id]
                if base_model_name == model_name:
                    base_model_path = model_manager.model_path[model_id]
                    base_model = model_manager.model[model_id]
                    print(f"    Adding patch model to {base_model_name} ({base_model_path})")
                    patched_model = load_single_patch_model_from_single_file(
                        state_dict, model_name, model_class, base_model, extra_kwargs, torch_dtype, device)
                    loaded_model_names.append(base_model_name)
                    loaded_models.append(patched_model)
                    model_manager.model.pop(model_id)
                    model_manager.model_path.pop(model_id)
                    model_manager.model_name.pop(model_id)
                    break
            else:
                break
    return loaded_model_names, loaded_models



class ModelDetectorTemplate:
    def __init__(self):
        pass

    def match(self, file_path="", state_dict={}):
        return False
    
    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        return [], []
    


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)


    def add_model_metadata(self, keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (model_names, model_classes, model_resource)


    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            loaded_model_names, loaded_models = load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)
            return loaded_model_names, loaded_models

        # Load models without strict matching
        # (the shape of parameters may be inconsistent, and the state_dict_converter will modify the model architecture)
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            loaded_model_names, loaded_models = load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)
            return loaded_model_names, loaded_models

        return loaded_model_names, loaded_models



class ModelDetectorFromSplitedSingleFile(ModelDetectorFromSingleFile):
    def __init__(self, model_loader_configs=[]):
        super().__init__(model_loader_configs)


    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                return True
        return False


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        # Split the state_dict and load from each component
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        valid_state_dict = {}
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                valid_state_dict.update(sub_state_dict)
        if super().match(file_path, valid_state_dict):
            loaded_model_names, loaded_models = super().load(file_path, valid_state_dict, device, torch_dtype)
        else:
            loaded_model_names, loaded_models = [], []
            for sub_state_dict in splited_state_dict:
                if super().match(file_path, sub_state_dict):
                    loaded_model_names_, loaded_models_ = super().load(file_path, valid_state_dict, device, torch_dtype)
                    loaded_model_names += loaded_model_names_
                    loaded_models += loaded_models_
        return loaded_model_names, loaded_models
    


class ModelDetectorFromHuggingfaceFolder:
    def __init__(self, model_loader_configs=[]):
        self.architecture_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)


    def add_model_metadata(self, architecture, huggingface_lib, model_name, redirected_architecture):
        self.architecture_dict[architecture] = (huggingface_lib, model_name, redirected_architecture)


    def match(self, file_path="", state_dict={}):
        if not isinstance(file_path, str) or os.path.isfile(file_path):
            return False
        file_list = os.listdir(file_path)
        if "config.json" not in file_list:
            return False
        with open(os.path.join(file_path, "config.json"), "r") as f:
            config = json.load(f)
        if "architectures" not in config and "_class_name" not in config:
            return False
        return True


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        with open(os.path.join(file_path, "config.json"), "r") as f:
            config = json.load(f)
        loaded_model_names, loaded_models = [], []
        architectures = config["architectures"] if "architectures" in config else [config["_class_name"]]
        for architecture in architectures:
            huggingface_lib, model_name, redirected_architecture = self.architecture_dict[architecture]
            if redirected_architecture is not None:
                architecture = redirected_architecture
            model_class = importlib.import_module(huggingface_lib).__getattribute__(architecture)
            loaded_model_names_, loaded_models_ = load_model_from_huggingface_folder(file_path, [model_name], [model_class], torch_dtype, device)
            loaded_model_names += loaded_model_names_
            loaded_models += loaded_models_
        return loaded_model_names, loaded_models
    


class ModelDetectorFromPatchedSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)


    def add_model_metadata(self, keys_hash_with_shape, model_name, model_class, extra_kwargs):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_name, model_class, extra_kwargs)


    def match(self, file_path="", state_dict={}):
        if not isinstance(file_path, str) or os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        return False


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, model_manager=None, **kwargs):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        loaded_model_names, loaded_models = [], []
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, extra_kwargs = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            loaded_model_names_, loaded_models_ = load_patch_model_from_single_file(
                state_dict, model_names, model_classes, extra_kwargs, model_manager, torch_dtype, device)
            loaded_model_names += loaded_model_names_
            loaded_models += loaded_models_
        return loaded_model_names, loaded_models



class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list: List[Preset_model_id] = [],
        downloading_priority: List[Preset_model_website] = ["ModelScope", "HuggingFace"],
        file_path_list: List[str] = [],
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        downloaded_files = download_models(model_id_list, downloading_priority) if len(model_id_list) > 0 else []
        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs),
            ModelDetectorFromSplitedSingleFile(model_loader_configs),
            ModelDetectorFromHuggingfaceFolder(huggingface_model_loader_configs),
            ModelDetectorFromPatchedSingleFile(patch_model_loader_configs),
        ]
        self.load_models(downloaded_files + file_path_list)


    def load_model_from_single_file(self, file_path="", state_dict={}, model_names=[], model_classes=[], model_resource=None):
        print(f"Loading models from file: {file_path}")
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        model_names, models = load_model_from_single_file(state_dict, model_names, model_classes, model_resource, self.torch_dtype, self.device)
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")


    def load_model_from_huggingface_folder(self, file_path="", model_names=[], model_classes=[]):
        print(f"Loading models from folder: {file_path}")
        model_names, models = load_model_from_huggingface_folder(file_path, model_names, model_classes, self.torch_dtype, self.device)
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")


    def load_patch_model_from_single_file(self, file_path="", state_dict={}, model_names=[], model_classes=[], extra_kwargs={}):
        print(f"Loading patch models from file: {file_path}")
        model_names, models = load_patch_model_from_single_file(
            state_dict, model_names, model_classes, extra_kwargs, self, self.torch_dtype, self.device)
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following patched models are loaded: {model_names}.")


    def load_lora(self, file_path="", state_dict={}, lora_alpha=1.0):
        if isinstance(file_path, list):
            for file_path_ in file_path:
                self.load_lora(file_path_, state_dict=state_dict, lora_alpha=lora_alpha)
        else:
            print(f"Loading LoRA models from file: {file_path}")
            is_loaded = False
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            for model_name, model, model_path in zip(self.model_name, self.model, self.model_path):
                for lora in get_lora_loaders():
                    match_results = lora.match(model, state_dict)
                    if match_results is not None:
                        print(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(model, state_dict, lora_prefix, alpha=lora_alpha, model_resource=model_resource)
                        is_loaded = True
                        break
            if not is_loaded:
                print(f"    Cannot load LoRA: {file_path}")


    def load_model(
        self,
        file_path,
        model_names=None,
        model_classes=None,
        model_resource=None,
        device=None,
        torch_dtype=None,
        model_kwargs=None,
    ):
        print(f"Loading models from: {file_path}")
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype
        manual_model_names = model_names or []
        manual_model_classes = model_classes or []
        manual_resource = model_resource
        manual_model_kwargs = model_kwargs or []
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path, torch_dtype=torch_dtype, device=device))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path, torch_dtype=torch_dtype, device=device)
        else:
            state_dict = None
        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                model_names, models = model_detector.load(
                    file_path, state_dict,
                    device=device, torch_dtype=torch_dtype,
                    allowed_model_names=model_names, model_manager=self,
                )
                for model_name, model in zip(model_names, models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                print(f"    The following models are loaded: {model_names}.")
                break
        else:
            if manual_model_names and manual_model_classes and state_dict is not None:
                if manual_resource is None:
                    manual_resource = "civitai"
                manual_kwargs = None
                if manual_model_kwargs:
                    manual_kwargs = [dict(kwargs) for kwargs in manual_model_kwargs]
                loaded_names, loaded_models = load_model_from_single_file(
                    state_dict,
                    manual_model_names,
                    manual_model_classes,
                    manual_resource,
                    torch_dtype,
                    device,
                    manual_extra_kwargs=manual_kwargs,
                )
                for model_name, model in zip(loaded_names, loaded_models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                if loaded_names:
                    print(f"    The following models are loaded: {loaded_names}.")
                else:
                    print(f"    No models were instantiated from override metadata for {file_path}.")
            else:
                print(f"    We cannot detect the model type. No models are loaded.")
        

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names, device=device, torch_dtype=torch_dtype)

    
    def fetch_model(self, model_name, file_path=None, require_model_path=False, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        else:
            if index is None:
                model = fetched_models[0]
                path = fetched_model_paths[0]
                print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[0]}.")
            elif isinstance(index, int):
                model = fetched_models[:index]
                path = fetched_model_paths[:index]
                print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[:index]}.")
            else:
                model = fetched_models
                path = fetched_model_paths
                print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths}.")
        if require_model_path:
            return model, path
        else:
            return model
        

    def to(self, device):
        for model in self.model:
            model.to(device)

