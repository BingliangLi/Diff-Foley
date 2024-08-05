#!/usr/bin/env python
# coding: utf-8

# ## Diff-Foley: Inference Pipeline.

# In[ ]:


from omegaconf import OmegaConf
import os
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.append("/".join(os.getcwd().split("/")[:-1]))
from diff_foley.util import instantiate_from_config


# ### 1. Loading Stage1 CAVP Model:

# In[3]:


import os
from demo_util import Extract_CAVP_Features 

# Set Device:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Default Setting:

fps = 4                                                     #  CAVP default FPS=4, Don't change it.
batch_size = 40   # Don't change it.
cavp_config_path = "./config/Stage1_CAVP.yaml"              #  CAVP Config
cavp_ckpt_path = "./diff_foley_ckpt/cavp_epoch66.ckpt"      #  CAVP Ckpt


# Initalize CAVP Model:
extract_cavp = Extract_CAVP_Features(fps=fps, batch_size=batch_size, device=device, config_path=cavp_config_path, ckpt_path=cavp_ckpt_path)


# ### 2. Loading Stage2 LDM Model:

# In[6]:


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


# In[7]:


# LDM Config:
ldm_config_path = "./config/Stage2_LDM.yaml"
ldm_ckpt_path = "./diff_foley_ckpt/ldm_epoch240.ckpt"
config = OmegaConf.load(ldm_config_path)

# Loading LDM:
latent_diffusion_model = load_model_from_config(config, ldm_ckpt_path)

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(21)

# ### 3. Data Preprocess

# In[20]:


# Sample1:
video_path = "./demo_videos/gun.mp4"
save_path = "./generate_samples/gun"
tmp_path = "./generate_samples/temp_folder" 
start_second = 0              # Video start second
truncate_second = 8.2         # Video end = start_second + truncate_second

# Extract Video CAVP Features & New Video Path:
cavp_feats, new_video_path = extract_cavp(video_path, start_second, truncate_second, tmp_path=tmp_path)


# In[21]:



# ### 4. Diff-Foley Generation:

# #### 4.(a) Double Guidance Load:

# In[22]:


# Whether use Double Guidance:
use_double_guidance = True

if use_double_guidance:
    classifier_config_path = "./config/Double_Guidance_Classifier.yaml"
    classifier_ckpt_path = "./diff_foley_ckpt/double_guidance_classifier.ckpt"
    classifier_config = OmegaConf.load(classifier_config_path)
    classifier = load_model_from_config(classifier_config, classifier_ckpt_path)
    


# In[23]:


from demo_util import inverse_op

sample_num = 1

# Inference Param:
cfg_scale = 4.5      # Classifier-Free Guidance Scale
cg_scale = 50        # Classifier Guidance Scale


steps = 25                # Inference Steps

sampler = "DPM_Solver"    # DPM-Solver Sampler
# sampler = "DDIM"        # DDIM Sampler
# sampler = "PLMS"        # PLMS Sampler


save_path = save_path + "_CFG{}_CG{}_{}_{}_useDG_{}".format(cfg_scale, cg_scale, sampler, steps, use_double_guidance)
os.makedirs(save_path, exist_ok=True)

# Video CAVP Features:
video_feat = torch.from_numpy(cavp_feats).unsqueeze(0).repeat(sample_num, 1, 1).to(device)
print(video_feat.shape)


# Truncate the Video Cond:
feat_len = video_feat.shape[1]
truncate_len = 32
window_num = feat_len // truncate_len


audio_list = []     # [sample_list1, sample_list2, sample_list3 ....]
for i in tqdm(range(window_num), desc="Window:"):
    start, end = i * truncate_len, (i+1) * truncate_len
    
    # 1). Get Video Condition Embed:
    embed_cond_feat = latent_diffusion_model.get_learned_conditioning(video_feat[:, start:end])     

    # 2). CFG unconditional Embedding:
    uncond_cond = torch.zeros(embed_cond_feat.shape).to(device)
    
    # 3). Diffusion Sampling:
    print("Using Double Guidance: {}".format(use_double_guidance))
    if use_double_guidance:
        audio_samples, _ = latent_diffusion_model.sample_log_with_classifier_diff_sampler(embed_cond_feat, origin_cond=video_feat, batch_size=video_feat.shape[0], sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=uncond_cond,classifier=classifier, classifier_guide_scale=cg_scale)  # Double Guidance
    else:
        audio_samples, _ = latent_diffusion_model.sample_log_diff_sampler(embed_cond_feat, batch_size=sample_num, sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=uncond_cond)           #  Classifier-Free Guidance
 
    # 4). Decode Latent:
    audio_samples = latent_diffusion_model.decode_first_stage(audio_samples)                     
    audio_samples = audio_samples[:, 0, :, :].detach().cpu().numpy()                               

    # 5). Spectrogram -> Audio:  (Griffin-Lim Algorithm)
    sample_list = []        #    [sample1, sample2, ....]
    for k in tqdm(range(audio_samples.shape[0]), desc="current samples:"):
        sample = inverse_op(audio_samples[k])
        sample_list.append(sample)
    audio_list.append(sample_list)


# In[24]:


# Save Samples:
path_list = []
for i in range(sample_num):      # sample_num
    current_audio_list = []
    for k in range(window_num):
        current_audio_list.append(audio_list[k][i])
    current_audio = np.concatenate(current_audio_list,0)
    print(current_audio.shape)
    sf.write(os.path.join(save_path, "sample_{}_diff.wav").format(i), current_audio, 16000)
    path_list.append(os.path.join(save_path, "sample_{}_diff.wav").format(i))
print("Gen Success !!")


# Concat The Video and Sound:
import subprocess
src_video_path = new_video_path
for i in range(sample_num):
    gen_audio_path = path_list[i]
    breakpoint()
    # out_path = os.path.join(save_path, "output_{}.mp4".format(i))
    out_audio_path = os.path.join(save_path, "output_audio_{}.wav".format(i))
    # cmd = ["ffmpeg" ,"-i" ,src_video_path,"-i" , gen_audio_path ,"-c:v" ,"copy" ,"-c:a" ,"aac" ,"-strict" ,"experimental", out_path]
    cmd = ["ffmpeg" ,"-i" , gen_audio_path , out_audio_path]
    subprocess.check_call(cmd)
print("Gen Success !!")





