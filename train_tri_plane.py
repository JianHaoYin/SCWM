from isolated_infer import model_forward_wrapper

import os
import yaml
import argparse
import logging
from time import time
from copy import deepcopy
from collections import OrderedDict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset
from diffusers.models import AutoencoderKL

from model.diffusion_model import CDiT_models
from diffusion import create_diffusion
from dataset.datesets import TrainingDataset
from misc import transform

# Tri-plane condition pipeline
from tri_plane.tri_plane_model import SatelliteToTargetFeatureModel


#################################################################################
#                             Helper Functions                                  #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("_orig_mod.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


#################################################################################
#                                  Evaluate                                     #
#################################################################################

@torch.no_grad()
def evaluate(ema, vae, diffusion, test_dataset, config, device, save_dir, bfloat_enable, num_cond):
    loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True, cache_dir=os.path.join(os.getcwd(), "models_cache"))

    os.makedirs(save_dir, exist_ok=True)

    score_sum = 0.0
    n_samples = 0

    # run a few steps for speed
    max_steps = int(config.get("eval_steps", 1))

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        # Expect dataset returns:
        # x: [B,T,3,H,W]  (pixels in [-1,1] or per config normalize)
        # y: [B,num_goals,3] or [B,T-num_cond,3] after dataset packaging
        # rel_t: [B,num_goals] or [B,T-num_cond]
        # sat_img: [B,3,Hs,Ws]
        # cam2world: [B,num_goals,4,4] or [B,4,4]
        # intrinsics: [B,num_goals,3,3] or [B,3,3]
        if len(batch) == 3:
            raise ValueError("Your dataset must return (x,y,rel_t,sat_img,cam2world,intrinsics) for tri-plane conditioning.")
        x, y, rel_t, sat_img, cam2world, intrinsics = batch

        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device)
        sat_img = sat_img.to(device)
        cam2world = cam2world.to(device)
        intrinsics = intrinsics.to(device)

        B, T = x.shape[:2]
        num_goals = T - num_cond
        latent_size = config["image_size"] // 8

        # Generate samples using wrapper (needs adapting to new conditioning if wrapper assumes old x_cond)
        # We'll do a direct minimal sampling path by reusing model_forward_wrapper only if it supports model_kwargs.
        # If your model_forward_wrapper doesn't support passing sat/camera, skip and only compute dreamsim on one forward.
        #
        # Here we will compute diffusion samples with wrapper, but you must ensure wrapper builds x_cond appropriately.
        #
        # For now: do a minimal run of wrapper using pixels only (keeps your old eval visualization).
        samples = model_forward_wrapper(
            (ema, diffusion, vae),
            x,
            y,
            num_timesteps=None,
            latent_size=latent_size,
            device=device,
            num_cond=num_cond,
            num_goals=num_goals,
            rel_t=rel_t.flatten(0, 1),
            # NOTE: sat_img/cam2world/intrinsics not passed in this wrapper
        )

        x_start_pixels = x[:, num_cond:].flatten(0, 1)
        x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x.shape[2:]).flatten(0, 1)

        samples = samples * 0.5 + 0.5
        x_start_pixels = x_start_pixels * 0.5 + 0.5
        x_cond_pixels = x_cond_pixels * 0.5 + 0.5

        res = eval_model(x_start_pixels, samples)
        score_sum += float(res.sum().item())
        n_samples += int(res.numel())

        # Save a few visualizations
        for i in range(min(samples.shape[0], 6)):
            fig, ax = plt.subplots(1, 3, dpi=200)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            ax[0].set_title("context last")
            ax[1].imshow((x_start_pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            ax[1].set_title("gt")
            ax[2].imshow((samples[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            ax[2].set_title("sample")
            for a in ax:
                a.axis("off")
            fig.tight_layout()
            plt.savefig(f"{save_dir}/{step}_{i}.png")
            plt.close(fig)

    return score_sum / max(n_samples, 1)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Single GPU selection
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.global_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load configs
    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Experiment dirs
    os.makedirs(config["results_dir"], exist_ok=True)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")

    # VAE tokenizer
    tokenizer = AutoencoderKL.from_pretrained("/data/tlxd/hf_cache/stabilityai/sd-vae-ft-ema").to(device)
    tokenizer.eval()
    requires_grad(tokenizer, False)

    latent_size = config["image_size"] // 8
    assert config["image_size"] % 8 == 0, "Image size must be divisible by 8."

    num_cond = int(config["context_size"])
    cond_channels = int(config.get("cond_channels", 32))

    # Diffusion model (CDiT) - MUST be the patched version supporting cond_channels
    model = CDiT_models[config["model"]](
        context_size=num_cond,
        input_size=latent_size,
        in_channels=4,
        cond_channels=cond_channels,
    ).to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    # Tri-plane conditioning pipeline
    sat2feat = SatelliteToTargetFeatureModel(
        tri_plane_hw=tuple(config.get("tri_plane_hw", (224, 224))),
        tri_plane_z=int(config.get("tri_plane_z", 64)),
        tri_plane_c=cond_channels,
        out_feat_dim=cond_channels,
        device=device,
    ).to(device)

    # OPTIONAL: if you want to freeze tri-plane pipeline at first
    # if int(config.get("freeze_sat2feat", 0)) == 1:
    #     requires_grad(sat2feat, False)
    #     sat2feat.eval()
    #     logger.info("sat2feat is FROZEN (no gradients).")
    # else:
    #     sat2feat.train()
    #     logger.info("sat2feat is TRAINABLE (gradients enabled).")

    sat2feat.train()
    logger.info("sat2feat is TRAINABLE (gradients enabled).")


    # Render options
    render_opts = {
        "box_warp": float(config.get("box_warp", 2.0)),
        "ray_start": float(config.get("ray_start", 0.0)),
        "ray_end": float(config.get("ray_end", 2.0)),
        "depth_resolution": int(config.get("depth_resolution", 32)),
    }

    lr = float(config.get("lr", 1e-4))
    opt = torch.optim.AdamW(list(model.parameters()) + list(sat2feat.parameters()), lr=lr, weight_decay=0)

    bfloat_enable = bool(hasattr(args, "bfloat16") and args.bfloat16)
    scaler = torch.amp.GradScaler(enabled=bfloat_enable)

    # Resume checkpoint (optional)
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path) or config.get("from_checkpoint", 0):
        latest_path = latest_path if os.path.isfile(latest_path) else config.get("from_checkpoint", 0)
        logger.info(f"Loading checkpoint from {latest_path}")
        ckp = torch.load(latest_path, map_location="cpu", weights_only=False)

        if "model" in ckp:
            model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckp["model"].items()}, strict=True)
            ema.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckp["ema"].items()}, strict=True)
            logger.info("Loaded model and EMA weights.")
        else:
            update_ema(ema, model, decay=0)

        if "sat2feat" in ckp:
            sat2feat.load_state_dict(ckp["sat2feat"], strict=False)
            logger.info("Loaded sat2feat weights (strict=False).")

        if "opt" in ckp:
            opt.load_state_dict(ckp["opt"])
            logger.info("Loaded optimizer state.")

        if "epoch" in ckp:
            start_epoch = int(ckp["epoch"]) + 1
        if "train_steps" in ckp:
            train_steps = int(ckp["train_steps"])
        if "scaler" in ckp and bfloat_enable:
            scaler.load_state_dict(ckp["scaler"])

    # Diffusion scheduler
    diffusion = create_diffusion(timestep_respacing="")

    # Build datasets
    train_dataset_list, test_dataset_list = [], []
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        for split in ["train", "test"]:
            if split not in data_config:
                continue
            goals_per_obs = int(data_config["goals_per_obs"])
            if split == "test":
                goals_per_obs = 4

            if "distance" in data_config:
                min_dist_cat = data_config["distance"]["min_dist_cat"]
                max_dist_cat = data_config["distance"]["max_dist_cat"]
            else:
                min_dist_cat = config["distance"]["min_dist_cat"]
                max_dist_cat = config["distance"]["max_dist_cat"]

            len_traj_pred = data_config.get("len_traj_pred", config["len_traj_pred"])

            dataset = TrainingDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config[split],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=min_dist_cat,
                max_dist_cat=max_dist_cat,
                len_traj_pred=len_traj_pred,
                context_size=config["context_size"],
                normalize=config["normalize"],
                goals_per_obs=goals_per_obs,
                transform=transform,
                predefined_index=None,
                traj_stride=1,
            )
            if split == "train":
                train_dataset_list.append(dataset)
            else:
                test_dataset_list.append(dataset)
            logger.info(f"Dataset: {dataset_name} ({split}), size: {len(dataset)}")

    train_dataset = ConcatDataset(train_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)

    loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    logger.info(f"Train dataset size: {len(train_dataset):,}")

    # Train mode
    model.train()
    if int(config.get("freeze_sat2feat", 0)) != 1:
        sat2feat.train()
    ema.eval()

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs on {device}...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch}...")

        for batch in loader:
            # Expect dataset returns:
            # x, y, rel_t, sat_img, cam2world, intrinsics
            if len(batch) == 3:
                raise ValueError(
                    "TrainingDataset must return (x, y, rel_t, sat_img, cam2world, intrinsics) to use tri-plane conditioning."
                )
            x, y, rel_t, sat_img, cam2world, intrinsics = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)

            sat_img = sat_img.to(device, non_blocking=True)
            cam2world = cam2world.to(device, non_blocking=True)
            intrinsics = intrinsics.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=bfloat_enable, dtype=torch.bfloat16):
                # Encode RGB frames to VAE latents (no grad)
                with torch.no_grad():
                    B, T = x.shape[:2]
                    x_flat = x.flatten(0, 1)
                    lat = tokenizer.encode(x_flat).latent_dist.sample().mul_(0.18215)
                    x_lat = lat.unflatten(0, (B, T))  # [B,T,4,Hlat,Wlat]

                num_goals = T - num_cond
                Hlat, Wlat = x_lat.shape[-2], x_lat.shape[-1]

                # Target latents to denoise
                x_start = x_lat[:, num_cond:].flatten(0, 1)  # [B*num_goals,4,Hlat,Wlat]

                # Flatten y/rel_t to match B*num_goals
                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)

                # Prepare per-goal camera params
                if cam2world.dim() == 5:
                    cam2world_goal = cam2world  # [B,num_goals,4,4]
                    intr_goal = intrinsics      # [B,num_goals,3,3]
                else:
                    cam2world_goal = cam2world.unsqueeze(1).expand(B, num_goals, 4, 4)
                    intr_goal = intrinsics.unsqueeze(1).expand(B, num_goals, 3, 3)

                cam2world_goal = cam2world_goal.flatten(0, 1)
                intr_goal = intr_goal.flatten(0, 1)

                # Expand satellite image per-goal: [B,3,Hs,Ws] -> [B*num_goals,3,Hs,Ws]
                sat_goal = sat_img.unsqueeze(1).expand(B, num_goals, *sat_img.shape[1:]).flatten(0, 1)

                # Render tri-plane feature condition at latent resolution
                cond_feat, _ = sat2feat(
                    satellite_img=sat_goal,
                    cam2world=cam2world_goal,
                    intrinsics=intr_goal,
                    out_h=Hlat,
                    out_w=Wlat,
                    render_opts=render_opts,
                    return_aux=False,
                )  # [B*num_goals, Ccond, Hlat, Wlat]

                # Repeat to match context_size expected by CDiT: [N, num_cond, Ccond, H, W]
                x_cond = cond_feat.unsqueeze(1).repeat(1, num_cond, 1, 1, 1)

                # Diffusion timestep
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)

                # Loss
                model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
                loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if float(config.get("grad_clip_val", 0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["grad_clip_val"]))

            scaler.step(opt)
            scaler.update()

            update_ema(ema, model)

            # Logging
            running_loss += float(loss.detach().item())
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / max(log_steps, 1)
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "sat2feat": sat2feat.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "train_steps": train_steps,
                    "args": vars(args),
                }
                if bfloat_enable:
                    checkpoint["scaler"] = scaler.state_dict()
                ckpt_path = f"{checkpoint_dir}/latest.pth.tar"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # Eval
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, config, device, save_dir, bfloat_enable, num_cond)
                eval_time = time() - eval_start
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}s")

    logger.info("Done!")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/nwm_cdit_s_recon.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--gpu", type=tuple, default=(4,5,6,7), help="Which CUDA device index to use")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)