from safetensors.torch import save_file
import gc
import numpy as np
import os
import torch
import einops
import traceback
import cv2
import logging
import json

import modules.async_worker as worker
from modules.util import generate_temp_filename, TimeIt, get_checkpoint_hashes, get_lora_hashes
from PIL import Image

import os
from comfy.model_base import LTXAV
from shared import path_manager, settings
import shared

from pathlib import Path
import random
from modules.pipeline_utils import (
    clean_prompt_cond_caches,
)

import comfy.utils
from comfy.sample import fix_empty_latent_channels
from comfy.sd import load_checkpoint_guess_config, load_state_dict_guess_config, VAE
from latent_preview import get_previewer
from tqdm import tqdm

#from comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF, GGUFModelPatcher, UnetLoaderGGUF
#from comfyui_gguf.ops import GGMLOps
from molbal_comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF, GGUFModelPatcher, UnetLoaderGGUF
from molbal_comfyui_gguf.ops import GGMLOps
#from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher, load_gguf_clip
#from calcuis_gguf.pig import DualClipLoaderGGUF as DualCLIPLoaderGGUF


from nodes import (
    CLIPTextEncode,
    DualCLIPLoader,
    VAEDecodeTiled,
    VAEDecode,
)

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, Noise_RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider, CFGGuider
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideo, LTXVConditioning, LTXVScheduler, LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_lt import ModelSamplingLTXV
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEDecode
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from comfy_extras.nodes_custom_sampler import BasicScheduler, BasicGuider
from comfy_extras.nodes_audio import VAEDecodeAudio


from comfy_extras.nodes_video import CreateVideo
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import Types

class pipeline:
    pipeline_type = ["video_pipepline"]

    class StableDiffusionModel:
        def __init__(self, clip, unet, vae, audio_vae=None):
            self.clip = clip
            self.unet = unet
            self.vae = vae
            self.audio_vae = audio_vae

        def to_meta(self):
            if self.unet is not None:
                self.unet.model.to("meta")
            if self.clip is not None:
                self.clip.cond_stage_model.to("meta")
            if self.vae is not None:
                self.vae.first_stage_model.to("meta")
            if self.audio_vae is not None:
                self.audio_vae.first_stage_model.to("meta")

    clip = None
    vae = None
    audio_vae = None
    model_hash = ""
    model_base = None
    model_hash_patched = ""
    model_base_patched = None
    conditions = None

    ggml_ops = GGMLOps()
    logger = logging.getLogger()

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = 1 + ((int(gen_data["image_number"] / 4.0) + 1) * 4)
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def get_clip_name(shortname):
        # List of short names and default names for different text encoders
        defaults = {
            "clip_t5": "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
            "clip_gemma3_12b": "gemma-3-12b-it-Q4_0.gguf",
            "clip_ltx23_text_proj": "ltx-2.3_text_projection_bf16.safetensors",
            "clip_qwen3vl_32b": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        }
        return settings.default_settings.get(shortname, defaults[shortname] if shortname in defaults else None)

    def get_vae_name(shortname):
        # List of short names and default names for different VAE's
        defaults = {
            "vae_ltxv_video": "hunyuanvideo15_vae_fp16.safetensors",
            "vae_ltxv23_audio": "LTX23_audio_vae_bf16.safetensors",
            "vae_ltxv23_video": "LTX23_video_vae_bf16.safetensors",
            "vae_minimax_h3_audio": "minimax_h3_audio_vae_fp32.safetensors",
            "vae_minimax_h3_video": "minimax_h3_video_vae_fp16.safetensors",
        }
        return settings.default_settings.get(shortname, defaults[shortname] if shortname in defaults else None)

    known_model_info = {
        "LTXV": {
            "latent": None,
            "clip_type": comfy.sd.CLIPType.LTXV,
            "clip_names": [get_clip_name("clip_t5")],
            "vae_name": get_vae_name("vae_ltxv_video"),
            "audio_vae_name": None,
        },
        "LTXAV": {
            "latent": None,
            "clip_type": comfy.sd.CLIPType.LTXV,
            "clip_names": [get_clip_name("clip_gemma3_12b"), get_clip_name("clip_ltx23_text_proj")],
            "vae_name": get_vae_name("vae_ltxv23_video"),
            "audio_vae_name": get_vae_name("vae_ltxv23_audio"),
            "frame_cnt": "((frames // 8) * 8) + 1",
            "flags": ["need_audio_latent"],
            "options": {"guider": "CFGGuider", "scheduler": "LTXVScheduler"}
        },
        "MiniMaxH3": {
            "latent": None,
            "clip_type": comfy.sd.CLIPType.MINIMAX,
            "clip_names": [get_clip_name("clip_qwen3vl_32b")],
            "vae_name": get_vae_name("vae_minimax_h3_video"),
            "audio_vae_name": get_vae_name("vae_minimax_h3_audio"),
            "frame_cnt": "((frames // 17) * 17) + 5",
            "options": {"fps": 24.0, "guider": "BasicGuider", "scheduler": "BasicScheduler"},
        },
    }

    def get_clip_and_vae(self, unet):
        unet_type = unet.model.__class__.__name__

        ret = self.known_model_info.get(unet_type, {})
        ret['unet_type'] = unet_type
        self.model_info = ret
        return ret

    def load_base_model(self, name, unet_only=False, input_unet=None, hash=None):
        if self.model_hash is not None and (self.model_hash == name or self.model_hash == hash):
            return

        self.model_base = None
        self.model_hash = None
        self.model_base_patched = None
        self.model_patched_hash = None
        self.conditions = None
        self.model_info = None

        default = None

        filename = shared.models.get_model_path(
            "checkpoints",
            name,
            hash=hash,
            default=default,
        )

        if filename is None:
            print(f"ERROR: Could not load checkpoint {name}")
            return

        if Path(filename).suffix == '.merge':
            print(f"Error: Model type not supported.")
            return

        if input_unet is None: # Be quiet if we already loaded a unet
            print(f"Loading base {'unet' if unet_only else 'model'}: {name}")

        gc.collect(generation=2)

        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        unet = None

        filename = str(filename) # FIXME use Path and suffix instead?
        if filename.endswith(".gguf") or unet_only:
            with torch.torch.inference_mode():
                try:
                    if filename.endswith(".gguf"):
                        try:
                            sd, extra = load_gguf_sd(filename)
                        except:
                            extra = {}
                            sd = load_gguf_sd(filename)

                        self.ggml_ops.Linear.dequant_dtype = None
                        self.ggml_ops.Linear.patch_dtype = None
                        self.logger.setLevel(logging.ERROR) # Supress error messages
                        unet = comfy.sd.load_diffusion_model_state_dict(
                            sd, model_options={"custom_operations": self.ggml_ops}, metadata=extra.get("metadata", {})
                        )
                        self.logger.setLevel(logging.WARNING)

                        unet = GGUFModelPatcher.clone(unet)
                        unet.patch_on_device = True
                    elif input_unet is not None:
                        if isinstance(input_unet, ModelPatcher):
                            unet = GGUFModelPatcher.clone(input_unet)
                            unet.patch_on_device = True
                        else:
                            unet = comfy.sd.load_diffusion_model_state_dict(
                                input_unet, model_options={"custom_operations": self.ggml_ops}
                            )
                            #unet = comfy.sd.load_diffusion_model_state_dict(input_unet)
                            try:
                                unet = GGUFModelPatcher.clone(unet)
                                unet.patch_on_device = True
                            except Exception as e:
                                unet = input_unet
                                print(f"ERROR: {e}")
                                traceback.print_exc()
                    else:
                        model_options = {}
                        model_options["dtype"] = torch.bfloat16 # FIXME should be a setting
                        unet = comfy.sd.load_diffusion_model(filename, model_options=model_options)

                    # Get text-encoders (clip) and vae to match the unet
                    model_info = self.get_clip_and_vae(unet)
                    self.model_info = model_info

                    # Special massaging of Lumina2 unet
# FIXME 
                    if model_info.get('model_sampling', None):
                        match model_info['model_sampling'][0]:
                            case 'AuraFlow':
                                unet = ModelSamplingAuraFlow().patch_aura(
                                    model=unet,
                                    shift=model_info['model_sampling'][1]
                                )[0]
                            case 'SD3':
                                unet = ModelSamplingSD3().patch(
                                    model=unet,
                                    shift=model_info['model_sampling'][1]
                                )[0]

                    # Load everything...
                    clip_paths = []
                    for clip_name in model_info['clip_names']:
                        clip_paths.append(
                            str(
                                path_manager.get_folder_file_path(
                                    "clip",
                                    clip_name,
                                    default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                                )
                            )
                        )

                    print(f"Loading CLIP: {model_info['clip_names']}")
                    if all(name.endswith(".safetensors") for name in clip_paths):
                        model_options = {}
                        device = comfy.model_management.get_torch_device()
                        if device == "cpu":
                            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                        clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=model_info['clip_type'], model_options=model_options)
                    else:
                        clip_loader = DualCLIPLoaderGGUF()
                        self.logger.setLevel(logging.ERROR) # Supress error messages
                        clip = clip_loader.load_patcher(
                            clip_paths,
                            model_info['clip_type'],
                            clip_loader.load_data(clip_paths)
                        )
                        self.logger.setLevel(logging.WARNING)


                    if model_info['vae_name'] == "pixel_space":
                        sd = {}
                        sd["pixel_space_vae"] = torch.tensor(1.0)
                    else:
                        vae_path = path_manager.get_folder_file_path(
                            "vae",
                            model_info['vae_name'],
                            default = os.path.join(path_manager.model_paths["vae_path"], model_info['vae_name'])
                        )
                        print(f"Loading VAE: {model_info['vae_name']}")
                        if str(vae_path).endswith(".gguf"):
                            sd = load_gguf_sd(str(vae_path))
                            metadata = None
                        else:
                            sd, metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
                    vae = comfy.sd.VAE(sd=sd, metadata=metadata)


                    if model_info['audio_vae_name'] in ["pixel_space", None]:
                        audio_vae = None
                    else:
                        audio_vae_path = path_manager.get_folder_file_path(
                            "vae",
                            model_info['audio_vae_name'],
                            default = os.path.join(path_manager.model_paths["vae_path"], model_info['audio_vae_name'])
                        )
                        print(f"Loading Audio VAE: {model_info['audio_vae_name']}")
                        if str(vae_path).endswith(".gguf"):
                            sd = load_gguf_sd(str(audio_vae_path))
                            metadata = None
                        else:
                            sd, metadata = comfy.utils.load_torch_file(str(audio_vae_path), return_metadata=True)

                        # https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/nodes.py#L2453C1-L2476C22
                        modeltype = model_info.get("unet_type", "")
                        match modeltype:
                            case "MiniMaxH3":
                                try:
                                    meta = metadata.get("minimax_h3_audio_vae", {})
                                    if isinstance(meta, str):
                                        meta = json.loads(meta)
                                    kwargs = {"minimax_h3_audio_vae": meta.get("kwargs", {})}
                                except:
                                    print(f"WARNING: unable to parse metadata: {metadata}")
                                    kwargs = {}
                                #audio_vae = VAE(sd=sd, metadata=kwargs)
                                metadata = {}
                                audio_vae = VAE(sd=sd, metadata=metadata)
                            case "LTXVA":
                                from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
                                audio_vae = AudioVAE(sd, metadata)
                            case _:
                                sd_audio = comfy.utils.state_dict_prefix_replace(
                                    dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True
                                )
                                audio_vae = VAE(sd=sd_audio, metadata=metadata)
                                audio_vae.throw_exception_if_invalid()

                    clip_vision = None
                except Exception as e:
                    unet = None
                    traceback.print_exc()

        else:
            try:
                with torch.torch.inference_mode():
                    unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)

                if clip == None or vae == None:
                    raise
            except:
                print(f"Trying to load as unet.")
                self.load_base_model(
                    filename,
                    unet_only=True
                )
                return

#            print(f"DEBUG: load aio?")
#            sd = None
#            unet = None
#            try:
#                with torch.torch.inference_mode():
#                    sd = comfy.utils.load_torch_file(filename)
#            except Exception as e:
#                # Failed loading
#                print(f"ERROR: Failed loading {filename}: {e}")
#
#            print(f"DEBUG: sd1: {type(sd)}")
#            try:
#                diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(sd.copy())
#                parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
#                if parameters == 0:
#                    sd = comfy.sd.load_diffusion_model(filename)
#            except:
#                sd = None
#                pass
#
#            print(f"DEBUG: sd2: {type(sd)}")
#            if sd is not None:
#                # Try to load as All-In-One checkpoint
#                try:
#                    aio = load_state_dict_guess_config(sd.copy())
#                    #aio = load_checkpoint_guess_config(sd.copy())
#                except Exception as e:
#                    print(f"DEBUG: error: {e}")
#                    aio = None
#                print(f"DEBUG: aio: {aio}")
#                if isinstance(aio, tuple):
#                    unet, clip, vae, clip_vision = aio
#
#                    if (
#                        isinstance(unet, ModelPatcher) and
#                        isinstance(clip, CLIP) and
#                        isinstance(vae, VAE)
#                    ):
#                        # If we got here, we have all models. Dump sd since we don't need it
#                        sd = None
#                    else:
#                        if isinstance(unet, ModelPatcher):
#                            sd = unet.clone()
#
#                if sd is not None:
#                    # We got something, assume it was a unet
#                    # Re-run load_base_model to get text-encoders and vae
#                    self.load_base_model(
#                        name,
#                        hash=hash,
#                        unet_only=True,
#                        input_unet=sd,
#                    )
#                    return
#

            else:
                unet = None

        if unet == None:
            print(f"Failed to load {name}")
            self.model_base = None
            self.model_hash = None
            self.model_base_patched = None
            self.model_patched_hash = None
        else:
            self.model_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, audio_vae=audio_vae
            )
            if not (self.model_base.unet.model.__class__.__name__ in self.known_model_info.keys()):
                print(
                    f"Model {self.model_base.unet.model.__class__.__name__} not supported. RuinedFooocus supports {list(self.known_model_info.keys())} models as video model."
                )
                self.model_base = None

            if self.model_base is not None:
                self.model_hash = hash if hash is not None else name
                self.model_base_patched = self.model_base
                self.model_patched_hash = None
                self.model_info = self.get_clip_and_vae(self.model_base_patched.unet)

        # Model Options
#        try:
#            self.model_info = self.get_clip_and_vae(self.xl_base.unet)
#        except:
#            self.model_info = {}
        # FIXME.. Remove?
#        options = self.model_info.get("options", {})
#        if options.get("ModelNoiseScale", None) is not None:
#            self.xl_base.unet = ModelNoiseScale().patch(
#                model=self.xl_base.unet,
#                noise_scale=options.get("ModelNoiseScale", 0.0),
#            )[0]
#        if options.get("HiDreamO1SeamSmoothing", False):
#            self.model_base.unet = HiDreamO1PatchSeamSmoothing().execute(
#                model=self.model_base.unet,
#                start_percent=0.80,
#                end_percent=1.00,
#                pattern='single_shift',
#                passes='ramp_2_4',
#                blend='median',
#                strength=1.00,
#            )[0]

        return

    def load_loras(self, loras):
        loaded_loras = []

        model = self.model_base

        for lora in loras:
            name = lora.get("name", "None")
            weight = lora.get("weight", 0)
            hash = lora.get("hash", None)
            if name == "None" or weight == 0:
                continue

            filename = shared.models.get_model_path(
                "loras",
                name,
                hash=hash,
            )

            if filename is None:
                continue

            print(f"Loading LoRAs: {name}")
            try:
                lora = comfy.utils.load_torch_file(filename, safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(
                    model.unet, model.clip, lora, weight, weight
                )
                model = self.StableDiffusionModel(
                    unet=unet,
                    clip=None,
                    vae=None,
                    audio_vae=None,
                )
                loaded_loras += [(name, weight)]
            except:
                pass
        self.model_base_patched = model
        self.model_hash_patched = str(loras)

        print(f"LoRAs loaded: {loaded_loras}")

        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.model_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        # Setup

        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)
        gen_data["frame_rate"] = float(self.model_info.get("options", {}).get("fps", settings.default_settings.get("video_fps", 30.0))) # Get fps from options, or settings
        frames = int(gen_data["original_image_number"] * gen_data["frame_rate"]) # Generate "Frame number" seconds of video
        frame_number = eval(self.model_info.get("frame_cnt", "frames")) # Massage the frames into something that match what the model require
        gen_data["width"] = (gen_data["width"] // 32) * 32 # FIXME size clipping should be a option per model type
        gen_data["height"] = (gen_data["height"] // 32) * 32

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Processing text encoding ...", "html/generate_video.jpeg")
            )

        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        clip_skip = 1

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            previewer = get_previewer(self.model_base_patched.unet.load_device, self.model_base_patched.unet.model.latent_format)

            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)

                y = (preview_buytes * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                y = einops.rearrange(y, 'b c t h w -> (b h) (t w) c')

                maxw = 1920
                maxh = 1080
                image = Image.fromarray(y)
                ow, oh = image.size
                scale = min(maxh / oh, maxw / ow)
                image = image.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)
            else:
                image = None

            status = "Generating video"
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (
                    int(100 * (step / total_steps)),
                    f"{status} - {step}/{total_steps}",
                    image
                )
            )
            pbar.update_absolute(step + 1, total_steps, None)

        # Get text_encoding

        with TimeIt("Text encoding"):
            print("Encoding prompts.")
            self.textencode("+", positive_prompt, clip_skip)
            self.textencode("-", negative_prompt, clip_skip)

        with TimeIt("Setting up latents"):
            print("Setting up latents and getting ready to sample.")
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Get initial latents ...", None)
            )

            # Video latent FIXME
            # i2v?
            if gen_data["input_image"]:
                image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                # FIXME: check model type here
                (positive, negative, video_latent) = LTXVImgToVideo().generate(
                    positive = self.conditions["+"]["cache"],
                    negative = self.conditions["-"]["cache"],
                    image = image,
                    vae = self.model_base_patched.vae,
                    width = gen_data["width"],
                    height = gen_data["height"],
                    length = frame_number,
                    batch_size = 1,
                    strength = 1,
                )
            else:
                positive = self.conditions["+"]["cache"]
                negative = self.conditions["-"]["cache"]
                modeltype = self.model_base_patched.unet.model.__class__.__name__
                match modeltype:
                    case "LTXAV":
                        video_latent = EmptyLTXVLatentVideo().generate(
                            width = gen_data["width"],
                            height = gen_data["height"],
                            length = frame_number,
                            batch_size = 1,
                        )[0]
                    case "MiniMaxH3":
                        video_latent = None

            audio_latent = None
            if "need_audio_latent" in self.model_info.get("flags", []):
                if self.model_info["audio_vae_name"] is not None:
                    # Audio latent FIXME
                    # FIXME: check model type here
                    audio_latent = LTXVEmptyLatentAudio().execute(
                        audio_vae = self.model_base_patched.audio_vae,
                        frames_number = frame_number,
                        frame_rate = gen_data["frame_rate"],
                        batch_size = 1,
                    )[0]

            if audio_latent is None:
                latent = video_latent
            else:
                # Combine audio and video
                latent = LTXVConcatAVLatent().execute(
                    video_latent = video_latent,
                    audio_latent = audio_latent,
                )[0]

        # Conditioning
        modeltype = self.model_base_patched.unet.model.__class__.__name__
        match modeltype:
            case "LTXAV":
                positive, negative = LTXVConditioning().execute(
                    positive = positive,
                    negative = negative,
                    frame_rate = gen_data["frame_rate"],
                )
            case "MiniMaxH3":
                positive, latent = MiniMaxH3ImageToVideo().execute(
                    clip = self.model_base_patched.clip,
                    vae = self.model_base_patched.vae,
                    prompt = positive_prompt,
                    width = gen_data["width"],
                    height = gen_data["height"],
                    length = frame_number,
                )
                # outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],

                negative = None
            case _:
                print(f"ERROR: Couldn't find conditioning for: {modeltype}")
                return

        # Sampler
        with TimeIt("Sampling"):
            print("Get sigmas.")

            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Getting simgas ...", None)
            )

            ksampler = KSamplerSelect().get_sampler(
                sampler_name = gen_data["sampler_name"],
            )[0]

            # Sigmas
            scheduler = self.model_info.get("options", {}).get("scheduler", "")
            match scheduler:
                case "LTXVScheduler":
                    sigmas = LTXVScheduler().execute(
                        steps = gen_data["steps"],
                        max_shift = 2.05,
                        base_shift = 0.95,
                        stretch = True,
                        terminal = 0.1,
                        latent = latent
                    )[0]
                case _:
                    sigmas = BasicScheduler().execute(
                        model = self.model_base_patched.unet,
                        scheduler = gen_data["scheduler"],
                        steps = gen_data["steps"],
                        denoise = 1.0,
                    )[0]
    
            # Guider
            opt_guider = self.model_info.get("options", {}).get("guider", "None")
            match opt_guider:
                case "CFGGuider":
                    guider = CFGGuider().execute(
                        model = self.model_base_patched.unet,
                        cfg = float(gen_data["cfg"]),
                        positive = positive,
                        negative = negative,
                    )[0]
                case "BasicGuider":
                    guider = BasicGuider().execute(
                        model = self.model_base_patched.unet,
                        conditioning = positive,
                    )[0]
                case _:
                    print(f"ERROR: Couldn't find guider: {opt_guider}")
                    return

            noise = Noise_RandomNoise(seed)
    
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Generating ...", None)
            )

            #
            # Sample
            #

        #denoised_output = SamplerCustomAdvanced().execute(
        #    noise = noise,
        #    guider = guider,
        #    sampler = ksampler,
        #    sigmas = sigmas,
        #    latent_image = latent,
        #)

            latent_image = latent["samples"]
            latent = latent.copy()
            latent_image = fix_empty_latent_channels(guider.model_patcher, latent_image, latent.get("downscale_ratio_spacial", None))
            latent["samples"] = latent_image

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]
    
            x0_output = {}
            #callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

            print("Sampling")
            samples = guider.sample(
                noise.generate_noise(latent),
                latent_image,
                ksampler,
                sigmas,
                denoise_mask=noise_mask,
                callback=callback_function,
                disable_pbar=False,
                seed=noise.seed,
            )
            samples = samples.to(comfy.model_management.intermediate_device())

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        # FIXME is this only for LTXV2?
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            denoised_output = latent.copy()
            denoised_output["samples"] = x0_out
        else:
            denoised_output = out

        match self.model_info['unet_type']:
            case 'LTXAV':
                if audio_latent is not None:
                    #video_latent, audio_latent = LTXVSeparateAVLatent().execute(
                    samples = LTXVSeparateAVLatent().execute(
                        av_latent = denoised_output,
                    )
                    video_samples = samples[0]
                    audio_samples = samples[1]
            case "LTXV":
#                out = latent.copy()
#                out.pop("downscale_ratio_spacial", None)
#                out["samples"] = samples
#                if "x0" in x0_output:
#                    x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
#                    if samples.is_nested:
#                        latent_shapes = [x.shape for x in samples.unbind()]
#                        x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
#                    denoised_output = latent.copy()
#                    denoised_output["samples"] = x0_out
#                else:
#                    denoised_output = out
# FIXME
                samples = [{"samples": samples}]
            case "MiniMaxH3":
                samples = denoised_output["samples"]
                video_samples = {"samples": samples.tensors[0]}
                if len(samples.tensors) >= 2:
                    audio_samples = {"samples": samples.tensors[1]}
                else:
                    audio_samples = None

        # Decode video

        print(f"VAE decode video.")
        decoded_latent = VAEDecodeTiled().decode(
            samples=video_samples,
            tile_size=512,
            overlap=64,
            temporal_size=4096,
            temporal_overlap=8,
            vae=self.model_base_patched.vae,
        )[0]


        if self.model_info['audio_vae_name'] is not None:
            # Decode audio
            print(f"VAE decode audio.")
#FIXME test ltxv2
#            audio = LTXVAudioVAEDecode().execute(
#                samples = samples[1],
#                audio_vae = self.model_base_patched.audio_vae,
#            )[0]
            try:
                audio = VAEDecodeAudio().execute(
                    samples = audio_samples,
                    vae = self.model_base_patched.audio_vae,
                )[0]
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                audio = None
        else:
            audio = None

        # Create Video
        video = CreateVideo().execute(
            images = decoded_latent,
            audio = audio,
            fps = gen_data["frame_rate"],
        )[0]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Saving ...", None)
            )

        filename = generate_temp_filename(
            folder=path_manager.model_paths["temp_outputs_path"], extension="tmp"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print("Saving video")
        # Save MP4
        codec = "auto"
        try:
            loras = []
            for lora_data in gen_data["loras"] if gen_data["loras"] is not None else []:
                if len(lora_data[0]) == 64 and all(c in '0123456789abcdefABCDEF' for c in lora_data[0]): # Looks like sha256?
                    hash = lora_data[0]
                else:
                    hash = None
                w, l  = lora_data[1].split(" - ", 1)
                if not l == "None":
                    loras.append({"name": l, "weight": float(w), "hash": hash})
            data = {
                "Prompt": gen_data["positive_prompt"],
                "Negative": gen_data["negative_prompt"],
                "steps": gen_data["steps"],
                "cfg": gen_data["cfg"],
                "width": gen_data["width"],
                "height": gen_data["height"],
                "seed": abs(int(gen_data["seed"])),
                "sampler_name": gen_data["sampler_name"],
                "scheduler": gen_data["scheduler"],
                "base_model_name": gen_data["base_model_name"],
                "base_model_hash": get_checkpoint_hashes(gen_data["base_model_name"])['SHA256'],
                "loras": [[f"{get_lora_hashes(lora['name'])['SHA256']}", f"{lora['weight']} - {lora['name']}"] for lora in loras],
                "software": "RuinedFooocus",
            }
        except:
            data = {"prompt": gen_data["positive_prompt"], "software": "RuinedFooocus"}
        metadata = {"metadata": json.dumps(data)}

        video.save_to(
            filename.with_suffix(".mp4"),
            format = Types.VideoContainer.MP4,
            codec = Types.VideoCodec(codec),
            metadata = metadata
        )

        pil_images = []
        for image in decoded_latent:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        # Save GIF
        # FIXME: scale down, gifs are too big
        compress_level=9 # Min = 0, Max = 9
        pil_images[0].save(
            filename.with_suffix(".gif"),
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0/gen_data["frame_rate"]),
            append_images=pil_images[1:],
            optimize=True,
            loop=0,
        )

        return [str(filename.with_suffix(".gif"))]
