import inspect
import os
from pathlib import Path
# from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer
from tqdm import tqdm
import numpy as np
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

# Some transformers versions do not expose VitsModel/VitsTokenizer.
# They are not needed for the FISC audio projector dimension check.
try:
    from transformers import VitsModel, VitsTokenizer
except ImportError:
    VitsModel = None
    VitsTokenizer = None

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass
from diffusers import AutoencoderKL
from diffusers import DDIMScheduler,DDIMInverseScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import Attention

from diffusers.utils.torch_utils import randn_tensor
try:
    from diffusers.pipeline_utils import DiffusionPipeline,AudioPipelineOutput
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline,AudioPipelineOutput
from diffusers import AudioLDM2Pipeline
from functools import partial
# if is_librosa_available():
import librosa
from diffusers.utils import logging,replace_example_docstring
import torchaudio
logger = logging.get_logger(__name__)  


class SpecifyGradient(torch.autograd.Function):
    """
    This code defines a custom gradient function using PyTorch's `torch.autograd.Function` class. It is particularly helpful when you want to manipulate gradients manually in a deep learning model that relies on automatic differentiation. The class is called `SpecifyGradient`, and contains two essential methods: `forward` and `backward`.

1. The `@staticmethod` decorator indicates that these are static methods and can be called on the class itself, without instantiating an object from the class.

2. The `forward` method takes two input arguments: `ctx` and `input_tensor`. `ctx` is a context object used to store information needed for backward computation. `input_tensor` is the input tensor to this layer in the neural network. The purpose of this method is to compute the forward pass and store any required information for the backward pass.

3. The `@custom_fwd` decorator is a user-defined decorator (not provided here) which presumably wraps or modifies the forward method in some way, most likely to add functionality like logging, error checking or other custom behavior.

4. Inside the `forward` method, the ground truth gradient `gt_grad` is saved using `ctx.save_for_backward()`. This stored information will be used later in the backward function. The forward function then returns a tensor of ones with the same device and data type as the input tensor. This tensor will be used in the backward pass as a scaling factor to adjust the gradients.

5. The `backward` method takes two input arguments: `ctx` and `grad_scale`. `ctx` is the same context object used in the forward pass. `grad_scale` is the gradient scaling factor used to adjust the gradients. The purpose of this method is to compute the gradient updates with respect to the input during backpropagation. 

6. The `@custom_bwd` decorator is another user-defined decorator (not provided here) which performs a similar role for the backward method as the `@custom_fwd` decorator does for the forward method.

7. Inside the `backward` method, the ground truth gradient `gt_grad` is retrieved from the saved tensors. It is then scaled by multiplying it with `grad_scale`. The method returns the scaled gradient `gt_grad` and `None`. The `None` value is returned because there are no gradients to compute for `gt_grad` with respect to the input tensor – it is assumed to be an external property that doesn't require gradient computation.

This custom gradient function can be used in situations where you need to have fine-grained control over the gradients in a neural network. For example, if you want to perform gradient clipping or apply noise to the gradients, you would use this `SpecifyGradient` function in place of a standard PyTorch layer.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def normalize_fisc_audio_feature(
    audio_feature,
    target_dim=32000,
    eps=1e-6,
    detach=True,
):
    """
    Normalize the audio feature used by FISC source/reference projectors.

    This function must be used consistently in training and inference.
    It intentionally only changes the feature passed into FISC. It should
    not be used for the latent passed into AudioLDM2 predict_noise().

    Args:
        audio_feature: Tensor shaped [B, ...] or [...]
        target_dim: flattened feature dimension expected by FISC projectors
        eps: numerical stability value
        detach: detach the feature from the upstream frozen audio encoder

    Returns:
        Tensor shaped [B, target_dim]
    """
    if audio_feature is None:
        return None

    if detach:
        audio_feature = audio_feature.detach()

    if audio_feature.dim() == 1:
        audio_feature = audio_feature.unsqueeze(0)

    audio_feature = audio_feature.float().flatten(1)
    current_dim = audio_feature.shape[1]

    if current_dim < target_dim:
        audio_feature = F.pad(audio_feature, (0, target_dim - current_dim), mode="constant", value=0.0)
    elif current_dim > target_dim:
        audio_feature = audio_feature[:, :target_dim]

    mean = audio_feature.mean(dim=1, keepdim=True)
    std = audio_feature.std(dim=1, keepdim=True).clamp_min(eps)
    audio_feature = (audio_feature - mean) / std
    audio_feature = torch.nan_to_num(audio_feature, nan=0.0, posinf=0.0, neginf=0.0)

    return audio_feature


@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor


class MyCrossAttnProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # save text-conditioned attention map only
        # get attention map of ref
        if hidden_states.shape[0] == 4: 
            attn.hs = hidden_states[2:3]
        # get attention map of trg
        else:
            attn.hs = hidden_states[1:2]

        return hidden_states

class PConLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches
        self.patch_size = patch_size


    def get_attn_pcon_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        if c == 256:
            res = int(res2/4)
            ref_noise_reshape = ref_noise.reshape(bs, res, 4, c).permute(0, 3, 1, 2)  # [B,C,T,F]
            trg_noise_reshape = trg_noise.reshape(bs, res, 4, c).permute(0, 3, 1, 2)
        elif c == 384:
            res = int(res2/2)
            ref_noise_reshape = ref_noise.reshape(bs, res, 2, c).permute(0, 3, 1, 2)  # [B,C,T,F] [1,384,64,2]
            trg_noise_reshape = trg_noise.reshape(bs, res, 2, c).permute(0, 3, 1, 2)
        elif c == 640:
            # In this case, frequency dim =1, we shuffle according to temporal dim
            res = int(res2/1)
            ref_noise_reshape = ref_noise.reshape(bs, res, 1, c).permute(0, 3, 1, 2) # [1,640,32,1] 
            trg_noise_reshape = trg_noise.reshape(bs, res, 1, c).permute(0, 3, 1, 2)
        else:
            print("[ERROR] Incorrect dim!")

        ref_noise_pooled = ref_noise_reshape
        trg_noise_pooled = trg_noise_reshape

        # Normalize feature dim
        ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1) # [1, 1280, 16, 16]
        trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

        ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1) # [1,T, F,C]

        patch_ids = np.random.permutation(ref_noise_pooled.shape[1])  # Random shuffle 256 channel index
        patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))] 
        patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

        ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1) # remove batch dim [B,T,F,C] -> [T,F,C]


        trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1) # [1,T, F,C]
        trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) # remove batch dim [B,T,F,C] -> [T,F,C]
        loss += self.PatchNCELoss(ref_sample, trg_sample).mean()  
        return loss
    
    def get_attn_cut_loss_org(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        
        ref_noise_reshape = ref_noise.permute(0, 2, 1)
        trg_noise_reshape = trg_noise.permute(0, 2, 1)

        # Down sample the attention maps
        for ps in self.patch_size:

            ref_noise_pooled = ref_noise_reshape
            trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1) 
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 1)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))] 
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 1)
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1)
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean()  
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T = 0.07):
        batch_size = batch_size # 1
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1] # F
        ref_noise = ref_noise.detach()
    
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) # [T,1]
        ref_noise = ref_noise.unsqueeze(0) # [1,T,F,C]
        trg_noise = trg_noise.unsqueeze(0) # [1,T,F,C]
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise.view(batch_size,npatches,-1), trg_noise.view(batch_size,npatches,-1).transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T
        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))
        return loss


class AudioLDM2_pipe(nn.Module):
    def __init__(self, device, fp16, vram_O, hf_key=None, t_range=[0.05, 0.95]): 
        """
        The `__init__` method initializes the class and loads a Stable Diffusion model using the specified version number, 
        and also sets the precision of the model to either float16 or float32 depending on the `fp16` parameter. 
        It sets the device the model will run on based on the `device` parameter. 
        If a `hf_key` parameter is provided, it will use the defined checkpoint, otherwise it will use 'cvssp/audioldm2'.
        """
        super().__init__()

        self.device = device

        print(f'[INFO] loading AudioLDM2 diffusion...')

        if fp16==True:
            self.precision_t = torch.float16
        else:
            self.precision_t = torch.float32

        local_audioldm2 = os.environ.get("AUDIOLDM2_LOCAL_PATH", "").strip()

        if hf_key is not None:
            print(f"[INFO] using personalized model ckpt: {hf_key}")
            self.repo_id = hf_key
            self.local_files_only = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("DIFFUSERS_OFFLINE"))
        else:
            if local_audioldm2:
                self.repo_id = local_audioldm2
                self.local_files_only = True
                print(f"[INFO] using local AudioLDM2: {self.repo_id}")
            else:
                self.repo_id = "cvssp/audioldm2"
                self.local_files_only = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("DIFFUSERS_OFFLINE"))
                print(f"[INFO] using pre defined cvssp/audioldm2, local_files_only={self.local_files_only}")

        if self.local_files_only and not Path(str(self.repo_id)).exists() and hf_key is None:
            raise FileNotFoundError(
                f"AudioLDM2 local path does not exist: {self.repo_id}. "
                "Set AUDIOLDM2_LOCAL_PATH or pass --audioldm2_path to your script."
            )

        pipe = AudioLDM2Pipeline.from_pretrained(
            self.repo_id,
            torch_dtype=self.precision_t,
            local_files_only=self.local_files_only,
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        # Prepare unet for extract attention map
        self.unet = self.prep_unet(self.unet)

        self.text_encoder_2 = pipe.text_encoder_2
        self.projection_model = pipe.projection_model
        self.language_model = pipe.language_model
        self.tokenizer_2 = pipe.tokenizer_2
        self.vocoder = pipe.vocoder
        self.feature_extractor = pipe.feature_extractor
        self.encode_prompt = pipe.encode_prompt
        self.mel_spectrogram_to_waveform = pipe.mel_spectrogram_to_waveform

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        # feature_extractor.requires_grad_(False)
        self.language_model.requires_grad_(False)
        self.projection_model.requires_grad_(False)
        self.vocoder.requires_grad_(False)


        self.scheduler = DDIMScheduler.from_pretrained(self.repo_id, subfolder="scheduler", torch_dtype=self.precision_t, local_files_only=self.local_files_only)
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(self.repo_id, subfolder="scheduler", torch_dtype=self.precision_t, local_files_only=self.local_files_only)
        self.scheduler_sampling = DDIMScheduler.from_pretrained(self.repo_id, subfolder="scheduler", torch_dtype=self.precision_t, local_files_only=self.local_files_only)
    

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) 
        # self.alphas = self.cosine_noise_schedule(self.num_train_timesteps).to(self.device)
        print(f'[INFO] loaded AudioLDM2 diffusion!')

    def cosine_noise_schedule(self, num_train_timesteps, s=0.008):
        """Generates a cosine noise schedule for diffusion models."""
        t = torch.linspace(0, num_train_timesteps, num_train_timesteps)  # Time steps
        f_t = torch.cos(((t / num_train_timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]  # Normalize to start at 1
        return alphas_cumprod

    def prep_unet(self, unet):
        for name, params in unet.named_parameters():
            if 'attn1' in name: # self-attention
                params.requires_grad = True
            else:
                params.requires_grad = False

        # replace the fwd function
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention":
                module.set_processor(MyCrossAttnProcessor())
        return unet

    @torch.no_grad()
    def get_text_embeds(self, prompt,use_guidance=True,negative_prompt = ""):
        """
            The `get_text_embeds` method takes a text prompt as input and returns
            the embeddings of that prompt using the `text_encoder`.
        """
        prompt_embds,attention_mask,generated_prompt_embds = self.encode_prompt(
                prompt = prompt,
                device = self.device,
                num_waveforms_per_prompt = 1,
                do_classifier_free_guidance= use_guidance,
                negative_prompt=negative_prompt)
        
        return prompt_embds,generated_prompt_embds,attention_mask

    def predict_noise(self, prompt_embds,generated_prompt_embds, attention_mask, mel_spec, 
                      guidance_scale=100, as_latent=True, t=None, noise=None, cfg=True,scheduler=None,disable_grad=True):
        if as_latent:
            latents = mel_spec
        else:
            latents = self.encode_audio(mel_spec)

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        if disable_grad ==True:
            with torch.no_grad():
                if noise is None:
                    # add noise
                    noise = torch.randn_like(latents)

                # x_t = \sqrt(\alpha_t)x_0 + \sqrt(1-\alpha_t) \eps, where \eps is the noise.
                # latent here is x_0
                if scheduler is None:
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                else:
                    latents_noisy = scheduler.add_noise(latents, noise, t)

                latent_model_input = torch.cat([latents_noisy] * 2)
                # Save input tensors for UNet
                noise_pred = self.unet(latent_model_input, t, 
                                    encoder_hidden_states=generated_prompt_embds,
                                    encoder_hidden_states_1=prompt_embds,
                                    encoder_attention_mask_1=attention_mask)[0]
        else:
            if noise is None:
                # add noise
                noise = torch.randn_like(latents)

            # x_t = \sqrt(\alpha_t)x_0 + \sqrt(1-\alpha_t) \eps, where \eps is the noise.
            # latent here is x_0
            if scheduler is None:
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            else:
                latents_noisy = scheduler.add_noise(latents, noise, t)
                

            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            noise_pred = self.unet(latent_model_input, t, 
                                encoder_hidden_states=generated_prompt_embds,
                                encoder_hidden_states_1=prompt_embds,
                                encoder_attention_mask_1=attention_mask)[0]
        if cfg==True:
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        return noise_pred, t, noise

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        """
        The `produce_latents` method takes text embeddings and a set of latents as inputs, 
        and produces the corresponding latents for the given text prompts using a generative model.
        """
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents


    def latents_to_audios(self, latents, return_mel=False):
        # if scale:
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spec = self.vae.decode(latents).sample
        audio = self.mel_spectrogram_to_waveform(mel_spec)
        if return_mel == True:
            return audio, mel_spec
        else:
            audio = self.mel_spectrogram_to_waveform(mel_spec)
            return audio


    def encode_audio(self,mels):

        posterior = self.vae.encode(mels).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

  
# =============================================================================
# FISC-SteerMusic 新增代码：反馈引导的指令语义补偿模块
# =============================================================================
# 该部分是根据“研究方案改进建议”加入的网络构建。
# 核心原则：
#   1. 不修改 AudioLDM2 backbone。
#   2. 不修改 SteerMusic 的 DDS/PDS 主公式。
#   3. 只在 target text condition 进入 denoiser 之前加入残差补偿：
#          c_tilde_tgt = c_t + Delta c_inst
#   4. 训练时只训练新增模块：
#          SourceAudioProjector, ReferenceAudioProjector,
#          ReferenceSemanticAdapter A_omega,
#          PromptCompensationNetwork P_phi,
#          FeedbackGate G_psi
# =============================================================================


class FISCAudioProjector(nn.Module):
    """源音频/参考音频投影器。

    关键修复：不要再用 nn.LazyLinear 自动推断输入维度。
    你的 FISC checkpoint 中 source_audio_projector.net.0.weight 是 (512, 32000)，
    所以推理时必须显式构建 Linear(32000, 512)。否则第一次 forward 如果吃到
    512 维特征，LazyLinear 会把 projector 固化成 (512, 512)，随后加载 checkpoint
    就会触发 (512, 32000) -> (512, 512) 的错误 resize。
    """
    def __init__(self, hidden_dim=512, audio_feature_dim=32000):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.audio_feature_dim = int(audio_feature_dim)
        self.net = nn.Sequential(
            nn.Linear(self.audio_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, audio_feature):
        # 中文注释：将 [B,C,T,F] 或其他形状的音频特征展平成 [B,D]。
        if audio_feature is None:
            return None
        if audio_feature.dim() == 1:
            audio_feature = audio_feature.unsqueeze(0)
        if audio_feature.dim() > 2:
            audio_feature = audio_feature.flatten(1)

        # 双保险：即使调用方忘了 normalize_fisc_audio_feature，也在 projector 内部
        # pad/truncate 到 checkpoint 训练时的固定维度。
        current_dim = int(audio_feature.shape[1])
        target_dim = int(self.audio_feature_dim)
        if current_dim < target_dim:
            audio_feature = F.pad(audio_feature, (0, target_dim - current_dim), mode="constant", value=0.0)
        elif current_dim > target_dim:
            audio_feature = audio_feature[:, :target_dim]

        return self.net(audio_feature)


class FISCReferenceSemanticAdapter(nn.Module):
    """Reference Semantic Adapter A_omega。

    方案对应：
        r^k = CrossAttn(Q=c_t, K=e_r, V=e_r)

    作用：
        target instruction 作为 query，从 reference audio feature 中选择
        和当前编辑目标相关的个性化音色/风格/空间感，而不是直接拼接全部参考信息。
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, target_token, reference_token, reference_mask=0.0):
        # 中文注释：非个性化模式 m_r=0，reference 分支关闭。
        if reference_token is None or float(reference_mask) == 0.0:
            return torch.zeros_like(target_token)
        q = target_token.unsqueeze(1)
        k = reference_token.unsqueeze(1)
        v = reference_token.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.squeeze(1)
        h = self.norm1(target_token + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h * float(reference_mask)


class FISCPromptCompensator(nn.Module):
    """Prompt Compensation Network P_phi + Feedback Gate G_psi。

    方案对应：
        Delta c_inst^k(t)
          = lambda^k(t) * P_phi(c_t, c_s, c_a, m_r e_r, tau_t, F^{k-1})

        c_tilde_tgt^k(t)
          = c_t + Delta c_inst^k(t)

    网络结构：
        P_phi：2-layer Transformer Adapter
        G_psi：3-layer MLP + sigmoid
        A_omega：ReferenceSemanticAdapter，个性化模式启用
    """
    def __init__(
        self,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        feedback_dim=6,
        time_dim=128,
        lambda_max=0.15,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feedback_dim = feedback_dim
        self.time_dim = time_dim
        self.lambda_max = lambda_max

        # 中文注释：将所有条件映射到统一 hidden_dim。
        self.ct_proj = nn.LazyLinear(hidden_dim)     # c_t，目标文本
        self.cs_proj = nn.LazyLinear(hidden_dim)     # c_s，源文本
        self.ca_proj = nn.LazyLinear(hidden_dim)     # c_a，源音频
        self.er_proj = nn.LazyLinear(hidden_dim)     # e_r，参考音频
        self.fb_proj = nn.Linear(feedback_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        # 中文注释：A_omega，target text query 从 reference audio 中检索相关语义。
        self.ref_adapter = FISCReferenceSemanticAdapter(hidden_dim, num_heads, dropout)

        # 中文注释：P_phi，轻量 Transformer Adapter。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # 中文注释：G_psi，输入 F^{k-1}, tau_t, m_r，输出补偿强度 lambda。
        self.gate = nn.Sequential(
            nn.Linear(feedback_dim + time_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 中文注释：由于 AudioLDM2 的两个 text condition 维度可能不同，
        # 这里使用 LazyLinear 自动适配输出维度。
        self.delta_to_text = nn.LazyLinear(1)

    @staticmethod
    def _pool(x):
        # 中文注释：把文本序列 [B,L,D] 池化成 [B,D] 条件 token。
        if x is None:
            return None
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() > 3:
            return x.flatten(1)
        return x

    @staticmethod
    def _expand(x, batch_size):
        if x is None:
            return None
        if x.shape[0] == batch_size:
            return x
        if x.shape[0] == 1:
            return x.expand(batch_size, *x.shape[1:])
        return x[:batch_size]

    @staticmethod
    def _time_embed(t, dim, device, dtype):
        # 中文注释：用 timestep embedding 近似 h_t，避免侵入 U-Net 内部。
        if t is None:
            t = torch.zeros(1, device=device)
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device)
        t = t.to(device=device, dtype=torch.float32).reshape(-1, 1)
        half = dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1)).reshape(1, -1)
        emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(dtype=dtype)

    def _make_delta_layer_if_needed(self, text_dim, device, dtype):
        # 中文注释：第一次 forward 时创建输出层，并零初始化，使未训练时不改变原模型。
        if isinstance(self.delta_to_text, nn.LazyLinear):
            self.delta_to_text = nn.Linear(self.hidden_dim, text_dim).to(device=device, dtype=dtype)
            nn.init.zeros_(self.delta_to_text.weight)
            nn.init.zeros_(self.delta_to_text.bias)

    def forward(
        self,
        target_embeds,
        source_embeds,
        source_audio_token,
        feedback,
        timestep,
        reference_token=None,
        reference_mask=0.0,
        apply_to_cond_only=True,
    ):
        dtype = target_embeds.dtype
        device = target_embeds.device
        batch_size, seq_len, text_dim = target_embeds.shape
        source_embeds = self._expand(source_embeds, batch_size)

        ct = self.ct_proj(self._pool(target_embeds))
        cs = self.cs_proj(self._pool(source_embeds))
        ca = self.ca_proj(self._expand(source_audio_token, batch_size).to(device=device, dtype=dtype))

        if reference_token is None:
            reference_token = torch.zeros_like(source_audio_token)
        er = self.er_proj(self._expand(reference_token, batch_size).to(device=device, dtype=dtype))

        if feedback is None:
            feedback = torch.zeros(batch_size, self.feedback_dim, device=device, dtype=dtype)
        if feedback.dim() == 1:
            feedback = feedback.unsqueeze(0)
        feedback = self._expand(feedback.to(device=device, dtype=dtype), batch_size)

        tau = self._time_embed(timestep, self.time_dim, device, dtype)
        tau = self._expand(tau, batch_size)
        mr = torch.full((batch_size, 1), float(reference_mask), device=device, dtype=dtype)

        # 中文注释：reference semantic adapter 输出 r^k。
        r = self.ref_adapter(ct, er, reference_mask=reference_mask)

        # 中文注释：P_phi 输入 token：Z=[c_t,c_s,c_a,r^k,F,tau_t]。
        tokens = torch.stack([ct, cs, ca, r, self.fb_proj(feedback), self.time_proj(tau)], dim=1)
        h = self.transformer(tokens)
        h_tgt = self.out_norm(h[:, 0])

        self._make_delta_layer_if_needed(text_dim, device, dtype)

        # 中文注释：生成 Delta c_inst 的方向。
        # 重要：这里不再允许 delta_vec 无限制放大，否则 FISC 很容易学成“过编辑放大器”。
        delta_vec = self.delta_to_text(h_tgt)
        delta_vec = F.layer_norm(delta_vec, delta_vec.shape[-1:])
        delta_vec = torch.clamp(delta_vec, min=-3.0, max=3.0)
        delta_vec = torch.nan_to_num(delta_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # 中文注释：G_psi 生成 lambda，并用 lambda_max 限制最大补偿幅度。
        lam = self.lambda_max * torch.sigmoid(self.gate(torch.cat([feedback, tau, mr], dim=-1)))
        lam = torch.nan_to_num(lam, nan=0.0, posinf=self.lambda_max, neginf=0.0)

        # 中文注释：将 Delta c_inst 广播到所有 text token。
        delta = (lam[:, None, :] * delta_vec[:, None, :]).expand(batch_size, seq_len, text_dim)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        # 中文注释：CFG 下只修改 conditional 分支，不修改 unconditional 分支。
        if apply_to_cond_only and batch_size >= 2:
            mask = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
            mask[batch_size // 2:] = 1.0
            delta = delta * mask

        return target_embeds + delta, delta, lam


class FISCModel(nn.Module):
    """完整 FISC 模型，包含改进方案中的所有新增网络。"""
    def __init__(
        self,
        hidden_dim=512,
        lambda_max=0.15,
        audio_feature_dim=32000,
        source_audio_dim=None,
        input_dim=None,
    ):
        super().__init__()

        # 兼容不同脚本里可能使用的参数名。最终只保留一个固定维度。
        if source_audio_dim is not None:
            audio_feature_dim = source_audio_dim
        if input_dim is not None:
            audio_feature_dim = input_dim

        self.hidden_dim = int(hidden_dim)
        self.audio_feature_dim = int(audio_feature_dim)
        self.lambda_max = float(lambda_max)

        # 中文注释：源音频编码投影，对应 c_a = W_s a_s。
        self.source_audio_projector = FISCAudioProjector(
            hidden_dim=self.hidden_dim,
            audio_feature_dim=self.audio_feature_dim,
        )

        # 中文注释：参考音频编码投影，对应 e_r = W_r E_r(x_r)。
        self.reference_audio_projector = FISCAudioProjector(
            hidden_dim=self.hidden_dim,
            audio_feature_dim=self.audio_feature_dim,
        )

        # 中文注释：AudioLDM2 有两个文本条件流，需要分别补偿。
        self.prompt_comp = FISCPromptCompensator(hidden_dim=self.hidden_dim, lambda_max=lambda_max)
        self.generated_comp = FISCPromptCompensator(hidden_dim=self.hidden_dim, lambda_max=lambda_max)

    def forward(
        self,
        target_prompt_embeds,
        target_generated_embeds,
        source_prompt_embeds,
        source_generated_embeds,
        source_audio_feature,
        feedback,
        timestep,
        reference_feature=None,
        reference_mask=0.0,
    ):
        ca = self.source_audio_projector(source_audio_feature)

        if reference_feature is None:
            er = torch.zeros_like(ca)
        else:
            er = self.reference_audio_projector(reference_feature)

        prompt_embeds, delta_prompt, lambda_prompt = self.prompt_comp(
            target_prompt_embeds, source_prompt_embeds, ca, feedback, timestep, er, reference_mask
        )
        generated_embeds, delta_generated, lambda_generated = self.generated_comp(
            target_generated_embeds, source_generated_embeds, ca, feedback, timestep, er, reference_mask
        )

        # 中文注释：L_reg = ||Delta c_inst||^2。
        reg = delta_prompt.pow(2).mean() + delta_generated.pow(2).mean()

        return {
            "prompt_embeds": prompt_embeds,
            "generated_prompt_embeds": generated_embeds,
            "delta_prompt": delta_prompt,
            "delta_generated": delta_generated,
            "lambda_prompt": lambda_prompt,
            "lambda_generated": lambda_generated,
            "reg": reg,
        }


def make_feedback_tensor(s_edit, s_pres, s_ref=0.0, prev_feedback=None, reference_mask=0.0, device=None, dtype=torch.float32):
    """构造连续反馈向量 F^k。"""
    if prev_feedback is None:
        d_edit, d_pres, d_ref = 0.0, 0.0, 0.0
    else:
        prev = prev_feedback.detach().flatten()
        d_edit = float(s_edit) - float(prev[0])
        d_pres = float(s_pres) - float(prev[1])
        d_ref = float(s_ref) - float(prev[2])
    return torch.tensor(
        [float(s_edit), float(s_pres), float(reference_mask) * float(s_ref), d_edit, d_pres, float(reference_mask) * d_ref],
        device=device,
        dtype=dtype,
    )


def latent_preservation_score(current_latent, source_latent):
    """源保持代理分数。正式实验可替换为 CQT1-PCC + TAC_q。"""
    return F.cosine_similarity(current_latent.flatten(1), source_latent.flatten(1), dim=-1).mean()


def noise_edit_score(noise_pred_tgt, noise_pred_src):
    """编辑达成代理分数。正式实验可替换为 CLAP audio-text similarity。"""
    diff = noise_pred_tgt - noise_pred_src
    return torch.tanh(diff.flatten(1).norm(dim=1).mean() / 100.0)


def over_edit_penalty(current_feedback, previous_feedback, eps=0.01):
    """过编辑惩罚 L_over。"""
    if previous_feedback is None:
        return torch.tensor(0.0, device=current_feedback.device, dtype=current_feedback.dtype)
    pres_drop = torch.relu(previous_feedback[1] - current_feedback[1])
    edit_gain = current_feedback[0] - previous_feedback[0]
    eps_tensor = torch.tensor(eps, device=current_feedback.device, dtype=current_feedback.dtype)
    return pres_drop * torch.relu(eps_tensor - edit_gain)


def save_fisc_checkpoint(fisc_model, path, extra=None):
    payload = {"fisc": fisc_model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def _strip_fisc_prefix_if_needed(state):
    """兼容 module. / fisc. 前缀。"""
    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("fisc."):
            k = k[len("fisc."):]
        clean[k] = v
    return clean


def _is_uninitialized_tensor(x):
    return isinstance(x, (UninitializedParameter, UninitializedBuffer))


def _safe_shape(x):
    if _is_uninitialized_tensor(x):
        return None
    if not hasattr(x, "shape"):
        return None
    try:
        return tuple(x.shape)
    except RuntimeError:
        return None


def _get_first_linear_in_features(module):
    if module is None:
        return None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return None


def load_fisc_checkpoint(fisc_model, ckpt_path, strict=False, map_location="cpu"):
    import torch

    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "fisc_state_dict" in ckpt:
            state = ckpt["fisc_state_dict"]
        elif "fisc" in ckpt and isinstance(ckpt["fisc"], dict):
            state = ckpt["fisc"]
        else:
            state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state = _strip_fisc_prefix_if_needed(state)
    model_state = fisc_model.state_dict()

    # 关键检查：audio projector 不允许 32000 -> 512 静默 resize。
    for key in [
        "source_audio_projector.net.0.weight",
        "reference_audio_projector.net.0.weight",
    ]:
        if key not in state or key not in model_state:
            continue

        ckpt_shape = _safe_shape(state[key])
        model_shape = _safe_shape(model_state[key])

        # checkpoint 里如果是未初始化 Lazy 参数，跳过检查，不要访问 .shape 崩溃
        if ckpt_shape is None or model_shape is None:
            print(f"[WARN] skip projector strict shape check for uninitialized tensor: {key}")
            continue

        if ckpt_shape != model_shape:
            model_dim = _get_first_linear_in_features(
                getattr(fisc_model, key.split(".")[0], None)
            )
            raise RuntimeError(
                "FISC audio projector dimension mismatch.\n"
                f"  key: {key}\n"
                f"  checkpoint shape: {ckpt_shape}\n"
                f"  current model shape: {model_shape}\n"
                f"  current model audio_feature_dim: {getattr(fisc_model, 'audio_feature_dim', None)}\n"
                f"  current projector in_features: {model_dim}\n"
                "请用 checkpoint 的输入维度构建 FISCModel，例如：\n"
                "  FISCModel(lambda_max=..., audio_feature_dim=32000)\n"
                "不要再对 projector 权重做 resize。"
            )

    filtered_state = {}
    skipped = []

    for k, v in state.items():
        if k not in model_state:
            skipped.append((k, "unexpected"))
            continue

        mv = model_state[k]
        ckpt_shape = _safe_shape(v)
        model_shape = _safe_shape(mv)

        if ckpt_shape is None:
            skipped.append((k, "checkpoint tensor is uninitialized"))
            continue
        if model_shape is None:
            skipped.append((k, "model tensor is uninitialized"))
            continue

        if ckpt_shape != model_shape:
            skipped.append((k, f"shape {ckpt_shape} != {model_shape}"))
            continue

        filtered_state[k] = v

    missing, unexpected = fisc_model.load_state_dict(filtered_state, strict=False)

    print("[INFO] loaded FISC checkpoint:", ckpt_path)
    print("[INFO] FISC model audio_feature_dim:", getattr(fisc_model, "audio_feature_dim", None))

    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)
    if skipped:
        print("[WARN] skipped checkpoint keys:")
        for item in skipped[:30]:
            print("   ", item)
        if len(skipped) > 30:
            print("   ... total skipped:", len(skipped))

    return fisc_model

def _strip_fisc_prefix_if_needed(state):
    """兼容 {'fisc': state_dict}、'module.' 前缀，以及 'fisc.' 前缀。"""
    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("fisc."):
            k = k[len("fisc."):]
        clean[k] = v
    return clean


def _get_first_linear_in_features(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return None


def load_fisc_checkpoint(fisc_model, ckpt_path, strict=False, map_location="cpu"):
    import torch

    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "fisc_state_dict" in ckpt:
            state = ckpt["fisc_state_dict"]
        elif "fisc" in ckpt:
            state = ckpt["fisc"]
        else:
            state = ckpt
    else:
        state = ckpt

    state = _strip_fisc_prefix_if_needed(state)
    model_state = fisc_model.state_dict()

    # 最关键的检查：这两个 projector 不允许自动 resize。只要不一致就直接报错，
    # 防止把 (512, 32000) 静默裁成 (512, 512)，导致实验结果失真。
    projector_keys = [
        "source_audio_projector.net.0.weight",
        "reference_audio_projector.net.0.weight",
    ]
    for key in projector_keys:
        if key in state and key in model_state:
            ckpt_shape = tuple(state[key].shape)
            model_shape = tuple(model_state[key].shape)
            if ckpt_shape != model_shape:
                model_dim = _get_first_linear_in_features(
                    getattr(fisc_model, key.split(".")[0], None)
                )
                raise RuntimeError(
                    "FISC audio projector dimension mismatch.\n"
                    f"  key: {key}\n"
                    f"  checkpoint shape: {ckpt_shape}\n"
                    f"  current model shape: {model_shape}\n"
                    f"  current model audio_feature_dim: {getattr(fisc_model, 'audio_feature_dim', None)}\n"
                    f"  current projector in_features: {model_dim}\n"
                    "请用 checkpoint 的输入维度构建 FISCModel，例如：\n"
                    "  FISCModel(lambda_max=..., audio_feature_dim=32000)\n"
                    "不要再对 projector 权重做 resize。"
                )

    filtered_state = {}
    skipped = []

    for k, v in state.items():
        if k not in model_state:
            skipped.append((k, "unexpected"))
            continue

        mv = model_state[k]

        # LazyLinear / Lazy modules may leave some tensors uninitialized.
        # Accessing .shape on them raises:
        # RuntimeError: Can't access the shape of an uninitialized parameter or buffer.
        # These tensors should be skipped instead of crashing checkpoint loading.
        if isinstance(v, (UninitializedParameter, UninitializedBuffer)):
            skipped.append((k, "checkpoint tensor is uninitialized"))
            continue
        if isinstance(mv, (UninitializedParameter, UninitializedBuffer)):
            skipped.append((k, "model tensor is uninitialized"))
            continue

        if tuple(v.shape) != tuple(mv.shape):
            skipped.append((k, f"shape {tuple(v.shape)} != {tuple(mv.shape)}"))
            continue

        filtered_state[k] = v

    missing, unexpected = fisc_model.load_state_dict(filtered_state, strict=strict)

    print("[INFO] loaded FISC checkpoint:", ckpt_path)
    print("[INFO] FISC model audio_feature_dim:", getattr(fisc_model, "audio_feature_dim", None))
    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)
    if skipped:
        print("[WARN] skipped mismatched/unexpected keys:")
        for item in skipped[:20]:
            print("   ", item)
        if len(skipped) > 20:
            print("   ... total skipped:", len(skipped))

    return fisc_model

# =========================
# DIMFIX FINAL OVERRIDE
# Override old load_fisc_checkpoint safely.
# =========================

def _dimfix_strip_fisc_prefix(state):
    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("fisc."):
            k = k[len("fisc."):]
        clean[k] = v
    return clean


def _dimfix_safe_shape(x):
    if x is None or not hasattr(x, "shape"):
        return None
    try:
        return tuple(x.shape)
    except RuntimeError as e:
        if "uninitialized" in str(e).lower():
            return None
        raise


def _dimfix_first_linear_in_features(module):
    if module is None:
        return None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return None


def load_fisc_checkpoint(fisc_model, ckpt_path, strict=False, map_location="cpu"):
    """
    Safe FISC checkpoint loader.

    1. 不再 resize source/reference audio projector 权重。
    2. 如果 checkpoint 里有 LazyLinear 未初始化参数，直接跳过。
    3. source_audio_projector 必须保持 checkpoint 的 32000 输入维度。
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "fisc_state_dict" in ckpt:
            state = ckpt["fisc_state_dict"]
        elif "fisc" in ckpt and isinstance(ckpt["fisc"], dict):
            state = ckpt["fisc"]
        else:
            state = ckpt
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state = _dimfix_strip_fisc_prefix(state)
    model_state = fisc_model.state_dict()

    projector_keys = [
        "source_audio_projector.net.0.weight",
        "reference_audio_projector.net.0.weight",
    ]

    for key in projector_keys:
        if key not in state or key not in model_state:
            continue

        ckpt_shape = _dimfix_safe_shape(state[key])
        model_shape = _dimfix_safe_shape(model_state[key])

        if ckpt_shape is None:
            print(f"[WARN] skip strict projector check because checkpoint tensor is uninitialized: {key}")
            continue

        if model_shape is None:
            print(f"[WARN] skip strict projector check because model tensor is uninitialized: {key}")
            continue

        if ckpt_shape != model_shape:
            model_dim = _dimfix_first_linear_in_features(
                getattr(fisc_model, key.split(".")[0], None)
            )
            raise RuntimeError(
                "FISC audio projector dimension mismatch.\n"
                f"  key: {key}\n"
                f"  checkpoint shape: {ckpt_shape}\n"
                f"  current model shape: {model_shape}\n"
                f"  current model audio_feature_dim: {getattr(fisc_model, 'audio_feature_dim', None)}\n"
                f"  current projector in_features: {model_dim}\n"
                "Do not resize projector weights. Build FISCModel with the checkpoint dim, e.g. audio_feature_dim=32000."
            )

    filtered_state = {}
    skipped = []

    for k, v in state.items():
        if k not in model_state:
            skipped.append((k, "unexpected"))
            continue

        ckpt_shape = _dimfix_safe_shape(v)
        model_shape = _dimfix_safe_shape(model_state[k])

        if ckpt_shape is None:
            skipped.append((k, "checkpoint tensor is uninitialized"))
            continue

        if model_shape is None:
            skipped.append((k, "model tensor is uninitialized"))
            continue

        if ckpt_shape != model_shape:
            skipped.append((k, f"shape {ckpt_shape} != {model_shape}"))
            continue

        filtered_state[k] = v

    missing, unexpected = fisc_model.load_state_dict(filtered_state, strict=False)

    print("[INFO] loaded FISC checkpoint:", ckpt_path)
    print("[INFO] FISC model audio_feature_dim:", getattr(fisc_model, "audio_feature_dim", None))

    if "source_audio_projector.net.0.weight" in filtered_state:
        print("[INFO] loaded source_audio_projector.net.0.weight:",
              tuple(filtered_state["source_audio_projector.net.0.weight"].shape))

    if missing:
        print("[WARN] missing keys:", missing)

    if unexpected:
        print("[WARN] unexpected keys:", unexpected)

    if skipped:
        print("[WARN] skipped checkpoint keys:")
        for item in skipped[:30]:
            print("   ", item)
        if len(skipped) > 30:
            print("   ... total skipped:", len(skipped))

    return fisc_model

# =========================
# END DIMFIX FINAL OVERRIDE
# =========================

# =========================
# DIMFIX SAFE SHAPE OVERRIDE
# This overrides the previous _dimfix_safe_shape.
# =========================
def _dimfix_safe_shape(x):
    """
    Safe shape reader for normal tensors and Lazy uninitialized tensors.

    Important:
    Do NOT use hasattr(x, "shape") here, because PyTorch UninitializedParameter
    raises RuntimeError when shape is accessed.
    """
    if x is None:
        return None

    try:
        from torch.nn.parameter import UninitializedParameter, UninitializedBuffer
        if isinstance(x, (UninitializedParameter, UninitializedBuffer)):
            return None
    except Exception:
        pass

    try:
        return tuple(x.shape)
    except RuntimeError as e:
        if "uninitialized" in str(e).lower():
            return None
        raise
    except Exception:
        return None

# =========================
# END DIMFIX SAFE SHAPE OVERRIDE
# =========================
