import os
import torch
import torch.nn.functional as F
import numpy as np
import gc

import folder_paths
import comfy.model_management as mm
import comfy.utils

try:
    import diffusers.models.activations
    def patch_geglu_inplace():
        """Patch GEGLU with inplace multiplication to save GPU memory."""
        def forward(self, hidden_states):
            hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden_states.mul_(self.gelu(gate))
        diffusers.models.activations.GEGLU.forward = forward
except:
    pass

from .pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt, tensor2vid

from .models.controlnext_vid_svd import ControlNeXtSDVModel
from .models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from .utils.scheduling_euler_discrete_karras_fix import EulerDiscreteScheduler as EulerDiscreteSchedulerKarras
from diffusers.schedulers import EulerDiscreteScheduler

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import AutoencoderKLTemporalDecoder

script_directory = os.path.dirname(os.path.abspath(__file__))


from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

class DownloadAndLoadControlNeXt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            
            "precision": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                    ], {
                        "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("CONTROLNEXT_PIPE",)
    RETURN_NAMES = ("controlnext_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "ControlNeXtWrapper"

    def loadmodel(self, precision):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = comfy.utils.ProgressBar(5)
        
        download_path = os.path.join(folder_paths.models_dir, "diffusers", "controlnext")
        unet_model_path = os.path.join(download_path, "controlnext-svd_v2-unet-fp16.safetensors")
        contolnet_model_path = os.path.join(download_path, "controlnext-svd_v2-controlnet-fp16.safetensors")
        
        if not os.path.exists(unet_model_path):
            log.info(f"Downloading model to: {unet_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/ControlNeXt-SVD-V2-Comfy", 
                                ignore_patterns=["*converted*"]
                                local_dir=download_path, 
                                local_dir_use_symlinks=False)

        log.info(f"Loading model from: {unet_model_path}")
        pbar.update(1)

        if not os.path.exists(svd_path):
            log.info(f"Downloading SVD model to: {svd_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1", 
                                allow_patterns=[f"*.json", "*fp16*"],
                                ignore_patterns=["*unet*"],
                                local_dir=svd_path, 
                                local_dir_use_symlinks=False)
        pbar.update(1)

        svd_path = os.path.join(folder_paths.models_dir, "diffusers", "stable-video-diffusion-img2vid-xt-1-1")

        unet_config = UNetSpatioTemporalConditionControlNeXtModel.load_config(os.path.join(script_directory, "configs", "unet_config.json"))
        log.info("Loading UNET")
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            self.unet = UNetSpatioTemporalConditionControlNeXtModel.from_config(unet_config)
        sd = comfy.utils.load_torch_file(os.path.join(unet_model_path))
        if is_accelerate_available:
            for key in sd:
                set_module_tensor_to_device(self.unet, key, dtype=dtype, device=device, value=sd[key])
        else:
            self.unet.load_state_dict(sd, strict=False)
        del sd
        pbar.update(1)

        log.info("Loading VAE")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae", variant="fp16", low_cpu_mem_usage=True).to(dtype).to(device).eval()

        log.info("Loading IMAGE_ENCODER")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_path, subfolder="image_encoder", variant="fp16", low_cpu_mem_usage=True).to(dtype).to(device).eval()
        pbar.update(1)
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(svd_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(svd_path, subfolder="feature_extractor")
        
        log.info("Loading ControlNeXt")
        self.controlnext = ControlNeXtSDVModel()
        self.controlnext.load_state_dict(comfy.utils.load_torch_file(os.path.join(contolnet_model_path)))
        self.controlnext = self.controlnext.to(dtype).to(device).eval()
       
        pipeline = StableVideoDiffusionPipelineControlNeXt(
            vae = self.vae, 
            image_encoder = self.image_encoder, 
            unet = self.unet, 
            scheduler = self.noise_scheduler,
            feature_extractor = self.feature_extractor, 
            controlnext=self.controlnext,
        )
        
        controlnextsvd_model = {
            'pipeline': pipeline,
            'dtype': dtype,
        }
        pbar.update(1)
        return (controlnextsvd_model,)
    

    
class ControlNextDiffusersScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "scheduler": (
                    [   
                        'EulerDiscreteScheduler',
                        'EulerDiscreteSchedulerKarras',
                        'EulerDiscreteScheduler_AYS',
                    ],
                    ), 
            "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 700.0, "step": 0.001}),
            "sigma_max": ("FLOAT", {"default": 700.0, "min": 0.0, "max": 700.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "loadmodel"
    CATEGORY = "ControlNeXtSVD"

    def loadmodel(self, scheduler, sigma_min, sigma_max):

        scheduler_config = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "interpolation_type": "linear",
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "set_alpha_to_one": False,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "timestep_type": "continuous",
            "trained_betas": None,
            "use_karras_sigmas": False
            }
        if scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
            sigmas = None
        elif scheduler == 'EulerDiscreteScheduler_AYS':
            noise_scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
            sigmas = [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]
        elif scheduler == 'EulerDiscreteSchedulerKarras':
            scheduler_config['use_karras_sigmas'] = True
            noise_scheduler = EulerDiscreteSchedulerKarras.from_config(scheduler_config)
            sigmas = None      
        
        scheduler_options = {
            "noise_scheduler": noise_scheduler,
            "sigmas": sigmas,
        }

        return (scheduler_options,)
        
class ControlNextSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controlnext_pipeline": ("CONTROLNEXT_PIPE",),
            "ref_image": ("IMAGE",),
            "pose_images": ("IMAGE",),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "motion_bucket_id": ("INT", {"default": 127, "min": 0, "max": 1000, "step": 1}),
            "cfg_min": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "cfg_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "fps": ("INT", {"default": 7, "min": 2, "max": 100, "step": 1}),
            "controlnext_cond_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 10.0, "step": 0.01}),
            "context_size": ("INT", {"default": 24, "min": 1, "max": 128, "step": 1}),
            "context_overlap": ("INT", {"default": 6, "min": 1, "max": 128, "step": 1}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),            
            },
            "optional": {
                "optional_scheduler": ("DIFFUSERS_SCHEDULER",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "ControlNeXtSVD"

    def process(self, controlnext_pipeline, ref_image, pose_images, cfg_min, cfg_max, controlnext_cond_scale, motion_bucket_id, steps, seed, noise_aug_strength, fps, keep_model_loaded, 
                context_size, context_overlap, optional_scheduler=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = controlnext_pipeline['dtype']
        pipeline = controlnext_pipeline['pipeline']

        original_scheduler = pipeline.scheduler

        if optional_scheduler is not None:
            log.info(f"Using optional scheduler: {optional_scheduler['noise_scheduler']}")
            pipeline.scheduler = optional_scheduler['noise_scheduler']
            sigmas = optional_scheduler['sigmas']

            if sigmas is not None and (steps + 1) != len(sigmas):
                sigmas = loglinear_interp(sigmas, steps + 1)
                sigmas = sigmas[-(steps + 1):]
                sigmas[-1] = 0
                log.info(f"Using timesteps: {sigmas}")
        else:
            pipeline.scheduler = original_scheduler
            sigmas = None
  
        B, H, W, C = pose_images.shape

        assert B >= context_size, "The number of poses must be greater than the context size"

        ref_image = ref_image.permute(0, 3, 1, 2)
        pose_images = pose_images.permute(0, 3, 1, 2)
        pose_images = pose_images * 2 - 1

        ref_image = ref_image.to(device).to(dtype)
        pose_images = pose_images.to(device).to(dtype)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        frames = pipeline(
            ref_image, 
            pose_images, 
            num_frames=B,
            frames_per_batch=context_size,
            overlap=context_overlap,
            motion_bucket_id=motion_bucket_id,
            min_guidance_scale=cfg_min,
            max_guidance_scale=cfg_max,
            controlnext_cond_scale=controlnext_cond_scale,
            height=H,
            width=W, 
            fps=fps,
            noise_aug_strength=noise_aug_strength, 
            num_inference_steps=steps,
            generator=generator,
            sigmas = sigmas,
            decode_chunk_size=2, 
            output_type="latent",
            return_dict="false",
            #device=device,
        ).frames

        if not keep_model_loaded:
            pipeline.unet.to(offload_device)
            pipeline.vae.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        return {"samples": frames},

class ControlNextDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controlnext_pipeline": ("CONTROLNEXT_PIPE",),
            "samples": ("LATENT",),
            "decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "ControlNeXtSVD"

    def process(self, controlnext_pipeline, samples, decode_chunk_size):
        mm.soft_empty_cache()
    
        pipeline = controlnext_pipeline['pipeline']
        num_frames = samples['samples'].shape[0]
        try:
            frames = pipeline.decode_latents(samples['samples'], num_frames, decode_chunk_size)
        except:
            frames = pipeline.decode_latents(samples['samples'], num_frames, 1)
        frames = tensor2vid(frames, pipeline.image_processor, output_type="pt")
        
        frames = frames.squeeze(1)[1:].permute(0, 2, 3, 1).cpu().float()

        return frames,

class ControlNextGetPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ref_image": ("IMAGE",),
            "pose_images": ("IMAGE",),
            "include_body": ("BOOLEAN", {"default": True}),
            "include_hand": ("BOOLEAN", {"default": True}),
            "include_face": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("poses_with_ref", "pose_images")
    FUNCTION = "process"
    CATEGORY = "ControlNextWrapper"

    def process(self, ref_image, pose_images, include_body, include_hand, include_face):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        from .dwpose.util import draw_pose
        from .dwpose.dwpose_detector import DWposeDetector

        assert ref_image.shape[1:3] == pose_images.shape[1:3], "ref_image and pose_images must have the same resolution"

        #yolo_model = "yolox_l.onnx"
        #dw_pose_model = "dw-ll_ucoco_384.onnx"
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

        if not os.path.exists(model_det):
            log.info(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            log.info(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model) 

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det)
            self.pose = torch.jit.load(model_pose)

            self.dwprocessor = DWposeDetector(
                model_det=self.det,
                model_pose=self.pose)
        
        ref_image = ref_image.squeeze(0).cpu().numpy() * 255

        self.det = self.det.to(device)
        self.pose = self.pose.to(device)

        # select ref-keypoint from reference pose for pose rescale
        ref_pose = self.dwprocessor(ref_image)
        #ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
        ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ref_keypoint_id = [i for i in ref_keypoint_id \
            #if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
            if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
        ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
 
        height, width, _ = ref_image.shape
        pose_images_np = pose_images.cpu().numpy() * 255

        # read input video
        pbar = comfy.utils.ProgressBar(len(pose_images_np))
        detected_poses_np_list = []
        for img_np in pose_images_np:
            detected_poses_np_list.append(self.dwprocessor(img_np))
            pbar.update(1)

        self.det = self.det.to(offload_device)
        self.pose = self.pose.to(offload_device)

        detected_bodies = np.stack(
            [p['bodies']['candidate'] for p in detected_poses_np_list if p['bodies']['candidate'].shape[0] == 18])[:,
                        ref_keypoint_id]
        # compute linear-rescale params
        ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
        fh, fw, _ = pose_images_np[0].shape
        ax = ay / (fh / fw / height * width)
        bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
        a = np.array([ax, ay])
        b = np.array([bx, by])
        output_pose = []
        # pose rescale 
        for detected_pose in detected_poses_np_list:
            if include_body:
                detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
            if include_hand:
                detected_pose['faces'] = detected_pose['faces'] * a + b
            if include_face:
                detected_pose['hands'] = detected_pose['hands'] * a + b
            im = draw_pose(detected_pose, height, width, include_body=include_body, include_hand=include_hand, include_face=include_face)
            output_pose.append(np.array(im))

        output_pose_tensors = [torch.tensor(np.array(im)) for im in output_pose]
        output_tensor = torch.stack(output_pose_tensors) / 255

        ref_pose_img = draw_pose(ref_pose, height, width, include_body=include_body, include_hand=include_hand, include_face=include_face)
        ref_pose_tensor = torch.tensor(np.array(ref_pose_img)) / 255
        output_tensor = torch.cat((ref_pose_tensor.unsqueeze(0), output_tensor))
        output_tensor = output_tensor.permute(0, 2, 3, 1).cpu().float()
        
        return output_tensor, output_tensor[1:]


class ControlNextSVDApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "pose_images": ("IMAGE",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "blocks": ("STRING",{"default": "3"}),
                    "input_block_patch_after_skip": ("BOOLEAN", {"default": True}),
                    }
                }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "patch"
    CATEGORY = "ControlNeXtSVD"

    def patch(self, model, pose_images, strength, blocks, input_block_patch_after_skip):

        device = mm.get_torch_device()
        dtype = mm.unet_dtype()

        B, H, W, C = pose_images.shape

        pose_images = pose_images.clone()
        pose_images = pose_images.permute(0, 3, 1, 2).unsqueeze(0)
        #pose_images = pose_images * 2 - 1
        #pose_images = pose_images.to(device).to(dtype)
        
        if not hasattr(self, 'controlnext'):
            self.controlnext = ControlNeXtSDVModel()
            self.controlnext.load_state_dict(comfy.utils.load_torch_file(os.path.join(script_directory, 'models', 'controlnext-svd_v2-controlnet-fp16.safetensors')))
            self.controlnext = self.controlnext.to(dtype).to(device).eval()

        block_list = [int(x) for x in blocks.split(',')] #for testing, blocks 0-3 possible to apply to, 3 after skip so far best

        def input_block_patch(h, transformer_options):
            if transformer_options['block'][1] in block_list and 0 in transformer_options["cond_or_uncond"]:

                sigma = transformer_options["sigmas"][0]
                
                log_sigma = sigma.log()
                min_log_sigma = torch.tensor(0.0002).log() 
                max_log_sigma = torch.tensor(700).log() #can I get these from the model?
                normalized_log_sigma = (log_sigma - min_log_sigma) / (max_log_sigma - min_log_sigma)             

                #AnimateDiff-Evolved context windowing, is this method slower than it should be?
                if "ad_params" in transformer_options and transformer_options["ad_params"]['sub_idxs'] is not None:
                    sub_idxs = transformer_options['ad_params']['sub_idxs']
                    controlnext_input = pose_images[:,sub_idxs].to(h.dtype).to(h.device).contiguous()

                    controlnext_input[:, 0, ...] = pose_images[:, 0, ...]
                else:
                    controlnext_input = pose_images.to(h.dtype).to(h.device)
                
                #print("controlnext_input shape: ", controlnext_input.shape) 
                #print("h shape: ", h.shape)

                conditional_controls = self.controlnext(controlnext_input, normalized_log_sigma)['output']

                mean_latents, std_latents = torch.mean(h, dim=(1, 2, 3), keepdim=True), torch.std(h, dim=(1, 2, 3), keepdim=True)
                mean_control, std_control = torch.mean(conditional_controls, dim=(1, 2, 3), keepdim=True), torch.std(conditional_controls, dim=(1, 2, 3), keepdim=True)
                conditional_controls = (conditional_controls - mean_control) * (std_latents / (std_control + 1e-5)) + mean_latents
                conditional_controls = F.adaptive_avg_pool2d(conditional_controls, h.shape[-2:])

                h = h + conditional_controls * 0.2 * strength
 
            return h
        model_clone = model.clone()
        if not input_block_patch_after_skip:
            model_clone.set_model_input_block_patch(input_block_patch)
        else:
            model_clone.set_model_input_block_patch_after_skip(input_block_patch)
           
        return (model_clone, )

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadControlNeXt": DownloadAndLoadControlNeXt,
    "ControlNextSampler": ControlNextSampler,
    "ControlNextDecode": ControlNextDecode,
    "ControlNextGetPoses": ControlNextGetPoses,
    "ControlNextDiffusersScheduler": ControlNextDiffusersScheduler,
    "ControlNextSVDApply": ControlNextSVDApply
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadControlNeXt": "(Down)Load ControlNeXt",
    "ControlNextSampler": "ControlNext Sampler",
    "ControlNextDecode": "ControlNext Decode",
    "ControlNextGetPoses": "ControlNext GetPoses",
    "ControlNextDiffusersScheduler": "ControlNext Diffusers Scheduler",
    "ControlNextSVDApply": "ControlNext SVD Apply"
}
