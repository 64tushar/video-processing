# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import urllib.request
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord

from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models.tokenizer import MyGPT2Tokenizer
from eval_narrator import decode_one


##get new arguments for the demo model

def get_args_parser():
    parser = argparse.ArgumentParser(description='LaVid training and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d'])
    parser.add_argument('--root', default='datasets/Ego4D/video_5min_chunks_288px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata', default='datasets/Ego4D/ego4d_train.pkl',
                        type=str, help='path to metadata file')
    parser.add_argument('--metadata-aux', default=None, nargs='+',
                        type=str, help='path to metadata file (auxiliary data with pseudo narrations)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    parser.add_argument('--narration-selection', default='random',
                        choices=['random', 'concat'],
                        type=str, help='selection strategy if multiple narrations per clip')
    parser.add_argument('--num-hard-neg', default=0, type=int, help='number of hard negatives per video')
    # Model
    parser.add_argument('--model', default='CLIP_OPENAI_TIMESFORMER_BASE', type=str)
    parser.add_argument('--norm-embed', action='store_true', help='norm text and visual embed if set True')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--load-visual-pretrained', default=None, type=str,
                        help='path to pretrained model (in1k/in21k/...)')
    parser.add_argument('--project-embed-dim', default=256, type=int, help='embed dim after projection')
    parser.add_argument('--use-cls-token', action='store_true', help='use feature at [CLS] if set True')
    parser.add_argument('--contrastive-use-vissl', action='store_true', help='use contrastive implementation in vissl')
    parser.add_argument('--gated-xattn', action='store_true', help='use gated x-attn in VCLM_GPT2')
    parser.add_argument('--random-init-gpt2', action='store_true', help='random initialize params of text decoder in VCLM_GPT2')
    parser.add_argument('--timesformer-gated-xattn', action='store_true', help='use gated x-attn in TimeSformer')
    parser.add_argument('--timesformer-freeze-space', action='store_true', help='freeze space part in TimeSformer')
    parser.add_argument('--drop-path-rate', default=0., type=float, help='DropPath rate')
    parser.add_argument('--freeze-visual-vclm', action='store_true', help='freeze the visual model in VCLM_GPT2')
    parser.add_argument('--freeze-visual-vclm-temporal', action='store_true', help='freeze the temporal part of visual model in VCLM_GPT2')
    parser.add_argument('--freeze-lm-vclm', action='store_true', help='freeze the lm in VCLM_GPT2')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    # Training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--temperature-init', default=0.07, type=float,
                        help='init. logit temperature for samples')
    parser.add_argument('--freeze-temperature', action='store_true',
                        help='freeze logit temperature')
    parser.add_argument('--pseudo-temperature-init', default=0.07, type=float,
                        help='init. logit temperature for pseudo-narrated samples')
    parser.add_argument('--freeze-pseudo-temperature', action='store_true',
                        help='freeze logit temperature (for pseudo-narrated samples)')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--clip-grad-type', default='norm', choices=['norm', 'value'])
    parser.add_argument('--clip-grad-value', default=None, type=float, help='')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=99, type=int)
    parser.add_argument('--eval-in-middle-freq', default=-1, type=int)
    parser.add_argument('--save-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--use-zero', action='store_true',
                        help='use ZeroRedundancyOptimizer to save memory')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use gradient checkpointing during training for significantly less GPU usage')
    parser.add_argument('--use-half', action='store_true', help='evaluate using half-precision')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    


def main(args):

    vr = decord.VideoReader(args.video_path)
    num_seg = 4
    frame_ids = get_frame_ids(0, len(vr), num_segments=num_seg, jitter=False)
    frames = video_loader_by_frames('./', args.video_path, frame_ids)

    ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
    ckpt_path = os.path.join('modelzoo/', ckpt_name)
    os.makedirs('modelzoo/', exist_ok=True)
    if not os.path.exists(ckpt_path):
        print('downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name), ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # instantiate the model, and load the pre-trained weights
    model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,      # we use model.eval() anyway
        freeze_visual_vclm=False,  # we use model.eval() anyway
        num_frames=4,
        drop_path_rate=0.
    )

    model2 = getattr(models, args.model)(
        crip_grand_type=args.crip_grand_type,
        freeze_pseudo_temperature=args.freeze_pseudo_temperature is not None,
        text_use_cls_token=args.use_cls_token,
        project_embed_dim=args.project_embed_dim,
        gated_xattn=args.gated_xattn,        
        timesformer_freeze_space=args.timesformer_freeze_space,
        freeze_lm_vclm=args.freeze_lm_vclm,
        freeze_visual_vclm=args.freeze_visual_vclm,
        freeze_visual_vclm_temporal=args.freeze_visual_vclm_temporal,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        temperature_init=args.temperature_init,
    )
    
    model.load_state_dict(state_dict, strict=True)
    if args.cuda:
        model.cuda()
    model.eval()

    # transforms on input frames
    crop_size = 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
    ])
    frames = val_transform(frames)
    frames = frames.unsqueeze(0)  # fake a batch dimension

    tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    with torch.no_grad():
        if args.cuda:
            frames = frames.cuda(non_blocking=True)
        image_features = model.encode_image(frames)
        generated_text_ids, ppls = model.generate(
            image_features,
            tokenizer,
            target=None,  # free-form generation
            max_text_length=77,
            top_k=None,
            top_p=0.95,   # nucleus sampling
            num_return_sequences=10,  # number of candidates: 10
            temperature=0.7,
            early_stopping=True,
        )

    for i in range(10):
        generated_text_str = decode_one(generated_text_ids[i], tokenizer)
        print('{}: {}'.format(i, generated_text_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila narrator demo')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--video-path', default='assets/3c0dffd0-e38e-4643-bc48-d513943dc20b_012_014.mp4', type=str, help='video path')
    args = parser.parse_args()
    main(args)
