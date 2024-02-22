import math
import os
import random
from collections import defaultdict
from datetime import datetime
from os.path import join, dirname

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from kornia.filters import gaussian_blur2d
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.regression import explained_variance
from tqdm import tqdm

from downsamplers import SimpleDownsampler, AttentionDownsampler
from layers import ImplicitFeaturizer, MinMaxScaler, ChannelNorm
from util import (norm as reg_norm, unnorm as reg_unorm, generate_subset,
                  midas_norm, midas_unnorm, pca, PCAUnprojector)
from featurizers.util import get_featurizer
from datasets.util import get_dataset, SlicedDataset
from losses import entropy, total_variation
from datasets.JitteredImage import JitteredImage, apply_jitter

torch.multiprocessing.set_sharing_strategy('file_system')


def mag(t):
    return t.square().sum(1, keepdim=True).sqrt()


class ExplicitUpsampler(torch.nn.Module):

    def __init__(self, size, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.dim = dim
        self.feats = torch.nn.Parameter(F.normalize(torch.randn(1, dim, size, size), dim=1))

    def forward(self, x):
        return self.feats


@hydra.main(config_path="configs", config_name="implicit_upsampler.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(0)

    input_size_h = 224
    input_size_w = 224 * 2

    final_size = 14
    proj_dim = 128
    steps = 1200

    if cfg.model_type in {"dino16", "vit", "clip", "midas"}:
        multiplier = 1
        featurize_batch_size = 64
        kernel_size = 29
    elif cfg.model_type == "dinov2":
        multiplier = 1
        featurize_batch_size = 64
        kernel_size = 29
        final_size = 16
    elif cfg.model_type in {"dino8"}:
        multiplier = 1
        featurize_batch_size = 64
        kernel_size = 8
        final_size = 28
    elif cfg.model_type == "deeplab":
        multiplier = 1
        featurize_batch_size = 16
        kernel_size = 35
        final_size = 28
        proj_dim = 64
    elif cfg.model_type == "resnet50":
        multiplier = 2
        final_size = 14
        kernel_size = 35
        featurize_batch_size = 16
        steps = 500
    else:
        raise ValueError(f"Unknown model type {cfg.model_type}")

    color_feats = True
    pca_batch = 50
    blur_attn = cfg.blur_attn
    mag_bound_weight = 0.0

    max_pad = 30
    use_flips = True
    max_zoom = 1.8

    blur_pin = 0.1

    mag_weight = cfg.mag_weight
    mag_tv_weight = cfg.mag_tv_weight

    outlier_detection = cfg.outlier_detection
    redo = False
    n_images = 3000
    n_freqs = 30

    downsampler_type = cfg.downsampler_type

    if downsampler_type == "attention":
        batch_size = 10
        inner_batch = 10
    else:
        batch_size = 10
        inner_batch = 10

    tv_weight = 0.0
    filter_ent_weight = 0.0
    param_type = "implicit"

    use_norm = True

    experiment_name = f"exp1"

    feat_dir = join(cfg.output_root, "feats", experiment_name, cfg.dataset, cfg.split, cfg.model_type)
    log_dir = join(cfg.output_root, "logs", experiment_name, cfg.dataset, cfg.split, cfg.model_type)

    model, _, dim = get_featurizer(cfg.model_type, activation_type=cfg.activation_type, output_root=cfg.output_root)

    if use_norm:
        model = torch.nn.Sequential(model, ChannelNorm(dim))

    model = model.cuda()

    if cfg.model_type == "midas":
        norm = midas_norm
        unnorm = midas_unnorm
    else:
        norm = reg_norm
        unnorm = reg_unorm

    def project(imgs):
        if multiplier > 1:
            imgs = F.interpolate(imgs, scale_factor=multiplier, mode="bilinear")
        return model(imgs)

    transform = T.Compose([
        T.Resize(input_size_h),
        T.CenterCrop((input_size_h, input_size_w)),
        T.ToTensor(),
        norm
    ])

    full_dataset = get_dataset(dataroot=cfg.pytorch_data_dir,
                               name=cfg.dataset,
                               split=cfg.split,
                               transform=transform,
                               target_transform=None,
                               include_labels=False)

    if "sample" in cfg.dataset:
        partition_size = 1
        dataset = full_dataset
    else:
        if cfg.split == "val":
            full_dataset = full_dataset
        elif cfg.split == "train":
            full_dataset = Subset(full_dataset, generate_subset(len(full_dataset), 5000))
        else:
            raise ValueError(f"Unknown dataset {cfg.dataset}")

        full_size = len(full_dataset)
        partition_size = math.ceil(full_size / cfg.total_partitions)
        dataset = SlicedDataset(
            full_dataset,
            int(cfg.partition * partition_size),
            int((cfg.partition + 1) * partition_size))
    loader = DataLoader(dataset, shuffle=False)

    for img_num, batch in enumerate(loader):
        original_image = batch["img"].cuda()
        output_location = join(feat_dir, "/".join(batch["img_path"][0].split("/")[-1:]).replace(".jpg", ".pth"))

        os.makedirs(dirname(output_location), exist_ok=True)
        if not redo and os.path.exists(output_location) and not cfg.dataset == "sample":
            print(f"Found {output_location}, skipping")
            continue
        else:
            print(f"Did not find {output_location}, computing")

        if cfg.summarize:
            writer = SummaryWriter(join(log_dir, str(datetime.now())))

        params = []
        dataset = JitteredImage(original_image, n_images, use_flips, max_zoom, max_pad)
        loader = DataLoader(dataset, featurize_batch_size)
        with torch.no_grad():
            transform_params = defaultdict(list)
            lr_feats = project(original_image)
            [red_lr_feats], fit_pca = pca([lr_feats], dim=9, use_torch_pca=True)

            jit_features = []
            for transformed_image, tp in tqdm(loader):
                for k, v in tp.items():
                    transform_params[k].append(v)
                jit_features.append(project(transformed_image).cpu())

            transform_params = {k: torch.cat(v, dim=0) for k, v in transform_params.items()}
            jit_features = torch.cat(jit_features, dim=0)

            unprojector = PCAUnprojector(jit_features[:pca_batch], proj_dim, lr_feats.device, use_torch_pca=True)

            jit_features = unprojector.project(jit_features)
            lr_feats = unprojector.project(lr_feats)

            lr_mags = mag(lr_feats)

        if param_type == "implicit":
            end_dim = proj_dim
            if color_feats:
                start_dim = 5 * n_freqs * 2 + 3
            else:
                start_dim = 2 * n_freqs * 2

            upsampler = torch.nn.Sequential(
                MinMaxScaler(),
                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
                ChannelNorm(start_dim),
                torch.nn.Dropout2d(p=.2),
                torch.nn.Conv2d(start_dim, end_dim, 1),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(p=.2),
                ChannelNorm(end_dim),
                torch.nn.Conv2d(end_dim, end_dim, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(end_dim, end_dim, 1),
            ).cuda()
        elif param_type == "explicit":
            upsampler = ExplicitUpsampler(input_size_h, proj_dim).cuda()
        else:
            raise ValueError(f"Unknown param type {param_type}")
        params.append({"params": upsampler.parameters()})

        if downsampler_type == "simple":
            downsampler = SimpleDownsampler(kernel_size, final_size)
        else:
            downsampler = AttentionDownsampler(proj_dim + 1, kernel_size, final_size, blur_attn).cuda()

        params.append({"params": downsampler.parameters()})

        if outlier_detection:
            with torch.no_grad():
                outlier_detector = torch.nn.Conv2d(proj_dim, 1, 1).cuda()
                outlier_detector.weight.copy_(outlier_detector.weight * .1)
                outlier_detector.bias.copy_(outlier_detector.bias * .1)

            params.append({"params": outlier_detector.parameters()})
            get_scale = lambda feats: torch.exp(outlier_detector(feats) + .1).clamp_min(.0001)
        else:
            get_scale = lambda feats: torch.ones(feats.shape[0], 1, feats.shape[2], feats.shape[2],
                                                 device=feats.device,
                                                 dtype=feats.dtype)

        optim = torch.optim.NAdam(params)

        for step in tqdm(range(steps), f"Image {img_num} of {partition_size}"):
            for i in range(batch_size // inner_batch):
                upsampler.train()
                downsampler.train()

                hr_feats = upsampler(original_image)
                hr_mag = mag(hr_feats)
                hr_both = torch.cat([hr_mag, hr_feats], dim=1)
                loss = 0.0

                target = []
                hr_feats_transformed = []
                for j in range(inner_batch):
                    idx = torch.randint(n_images, size=())
                    target.append(jit_features[idx].unsqueeze(0))
                    selected_tp = {k: v[idx] for k, v in transform_params.items()}
                    hr_feats_transformed.append(apply_jitter(hr_both, selected_tp))

                target = torch.cat(target, dim=0).cuda(non_blocking=True)
                hr_feats_transformed = torch.cat(hr_feats_transformed, dim=0)

                output_both = downsampler(hr_feats_transformed, None)
                magnitude = output_both[:, 0:1, :, :]
                output = output_both[:, 1:, :, :]

                scales = get_scale(target)

                rec_loss = ((1 / (2 * scales ** 2)) * (output - target).square() + scales.log()).mean()

                loss += rec_loss

                if isinstance(downsampler, SimpleDownsampler):
                    filter_ent = entropy(downsampler.get_kernel())
                else:
                    filter_ent = 0.0

                if mag_bound_weight > 0.0:
                    mag_bound_loss = (hr_mag.max() - lr_mags.max()).square() + \
                                     (hr_mag.min() - lr_mags.min()).square()
                    loss += mag_bound_weight * mag_bound_loss

                if tv_weight > 0.0:
                    tv_loss = total_variation(hr_feats)
                    loss += tv_weight * tv_loss

                if filter_ent_weight > 0.0:
                    loss -= filter_ent_weight * filter_ent

                if mag_weight > 0.0:
                    mag_loss = (magnitude - mag(target)).square().mean()
                    mag_loss2 = (mag(output) - mag(target)).square().mean()
                    loss += mag_loss * mag_weight

                if mag_tv_weight > 0.0:
                    mag_tv = total_variation(hr_mag)
                    loss += mag_tv_weight * mag_tv

                if blur_pin > 0.0:
                    blur_pin_loss = (gaussian_blur2d(hr_feats, 5, (1.0, 1.0)) - hr_feats).square().mean()
                    loss += blur_pin * blur_pin_loss

                loss.backward()

                should_log = cfg.summarize and (i == (batch_size // inner_batch - 1))

                if should_log and step % 10 == 0:
                    upsampler.eval()
                    downsampler.eval()

                    writer.add_scalar("loss", loss, step)
                    mean_mae = (lr_feats.mean(dim=[2, 3]) - hr_feats.mean(dim=[2, 3])).abs().mean()
                    writer.add_scalar("mean_mae", mean_mae, step)
                    writer.add_scalar("rec loss", rec_loss, step)
                    writer.add_scalar("filter ent", filter_ent, step)
                    writer.add_scalar("mean scale", scales.mean(), step)

                    if mag_weight > 0.0:
                        writer.add_scalar("mag loss", mag_loss, step)
                        writer.add_scalar("mag loss2", mag_loss2, step)

                    if mag_tv_weight > 0.0:
                        writer.add_scalar("mag tv", mag_tv, step)

                    if tv_weight > 0.0:
                        writer.add_scalar("tv loss", tv_loss, step)

                    if mag_bound_weight > 0.0:
                        writer.add_scalar("mag_bound loss", mag_bound_loss, step)

                    if blur_pin > 0.0:
                        writer.add_scalar("blur pin loss", blur_pin_loss, step)

                if should_log and step % 100 == 0:
                    with torch.no_grad():
                        upsampler.eval()
                        downsampler.eval()

                        hr_feats = upsampler(original_image)
                        hr_mag = mag(hr_feats)
                        hr_both = torch.cat([hr_mag, hr_feats], dim=1)
                        target = []
                        hr_feats_transformed = []
                        for j in range(inner_batch):
                            idx = torch.randint(n_images, size=())
                            target.append(jit_features[idx].unsqueeze(0))
                            selected_tp = {k: v[idx] for k, v in transform_params.items()}
                            hr_feats_transformed.append(apply_jitter(hr_both, selected_tp))

                        target = torch.cat(target, dim=0).cuda(non_blocking=True)
                        hr_feats_transformed = torch.cat(hr_feats_transformed, dim=0)

                        output_both = downsampler(hr_feats_transformed, None)
                        output = output_both[:, 1:, :, :]
                        scales = get_scale(target)
                        big_target = unprojector(target)
                        big_output = unprojector(output)

                        ev = explained_variance(
                            big_output.flatten(),
                            big_target.flatten())
                        writer.add_scalar("explained_variance", ev, step)

                        [red_hr_feats, red_target, red_output], _ = pca([
                            unprojector(hr_feats), big_target, big_output
                        ], fit_pca=fit_pca, dim=9)

                        def up(x):
                            return F.interpolate(x.unsqueeze(0), hr_feats.shape[2:], mode="nearest").squeeze(0)

                        writer.add_image("feats/1/hr", red_hr_feats[0, :3], step)
                        writer.add_image("feats/2/hr", red_hr_feats[0, 3:6], step)
                        writer.add_image("feats/3/hr", red_hr_feats[0, 6:9], step)

                        np_arr = (red_lr_feats[0, :3].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8)
                        Image.fromarray(np_arr.detach().cpu().numpy()).save("../sample-images/low_res_feats.png")

                        writer.add_image("feats/1/lr", up(red_lr_feats[0, :3]), step)
                        writer.add_image("feats/2/lr", up(red_lr_feats[0, 3:6]), step)
                        writer.add_image("feats/3/lr", up(red_lr_feats[0, 6:9]), step)
                        writer.add_image("feats/1/pred", red_target[0, :3], step)
                        writer.add_image("feats/1/true", red_output[0, :3], step)
                        writer.add_image("image/original", unnorm(original_image)[0], step)
                        writer.add_image("image/transformed", unnorm(transformed_image)[0], step)

                        norm_scales = scales[0]
                        norm_scales /= scales.max()
                        writer.add_image("scales", norm_scales, step)
                        writer.add_histogram("scales hist", scales, step)

                        hr_lr_feats = F.interpolate(lr_feats, size=(input_size_h, input_size_w))
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        plt1 = axes[0].imshow(mag(hr_feats)[0, 0].detach().cpu())
                        plt.colorbar(plt1)
                        plt2 = axes[1].imshow(mag(hr_lr_feats)[0, 0].detach().cpu())
                        plt.colorbar(plt2)
                        writer.add_figure("magnitudes", fig, step)

                        if isinstance(downsampler, SimpleDownsampler):
                            prepped_filter = downsampler.get_kernel().squeeze().unsqueeze(0)
                            prepped_filter /= prepped_filter.max()
                            prepped_filter = (prepped_filter * 255).clamp(0, 255).to(torch.uint8)
                            writer.add_image("down/filter", prepped_filter, step)

                        if isinstance(downsampler, AttentionDownsampler):
                            prepped_att = downsampler.forward_attention(hr_both, None)[0]
                            prepped_att -= prepped_att.min()
                            prepped_att /= prepped_att.max()
                            prepped_att = (prepped_att * 255).clamp(0, 255).to(torch.uint8)
                            writer.add_image("down/att", prepped_att, step)

                            prepped_filter = downsampler.w.clone().squeeze()
                            prepped_filter -= prepped_filter.min()
                            prepped_filter /= prepped_filter.max()
                            prepped_filter = (prepped_filter * 255).clamp(0, 255).to(torch.uint8)
                            writer.add_image("down/w", prepped_filter.unsqueeze(0), step)

                            prepped_filter = downsampler.b.clone().squeeze()
                            prepped_filter -= prepped_filter.min()
                            prepped_filter /= prepped_filter.max()
                            prepped_filter = (prepped_filter * 255).clamp(0, 255).to(torch.uint8)
                            writer.add_image("down/b", prepped_filter.unsqueeze(0), step)

                    writer.flush()

            optim.step()
            optim.zero_grad()

        torch.save({"model": upsampler, "unprojector": unprojector}, output_location)


if __name__ == "__main__":
    my_app()
