#from isolated_infer import model_forward_wrapper
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

from pathlib import Path
ROOT = Path(__file__).resolve().parent

from model.diffusion_model import CDiT_models
from diffusion import create_diffusion
from dataset.datesets import TrainingDataset,SecondFrameTrainingDataset
from misc import transform

# Tri-plane condition pipeline
from tri_plane.tri_plane_model import SatelliteToTargetFeatureModel
from model.diffusion_model import TriPlaneCDiT   # 路径按你放置的位置改

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
    with open(ROOT / "config" / "eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(ROOT / args.config, "r") as f:
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


    # 1) base CDiT (must be the new CDiT supporting cond_channels)
    base_cdit = CDiT_models[config['model']](
        context_size=num_cond,
        input_size=latent_size,
        in_channels=4,
        cond_channels=cond_channels,
    ).to(device)

    # 2) wrap with TriPlaneCDiT: sat->triplane->render inside model.forward()
    render_opts = {
        "box_warp": float(config.get("box_warp", 2.0)),
        "ray_start": float(config.get("ray_start", 0.0)),
        "ray_end": float(config.get("ray_end", 2.0)),
        "depth_resolution": int(config.get("depth_resolution", 32)),
    }

    model = TriPlaneCDiT(
        cdit=base_cdit,
        context_size=num_cond,
        cond_channels=cond_channels,
        tri_plane_hw=tuple(config.get("tri_plane_hw", (224, 224))),
        tri_plane_z=int(config.get("tri_plane_z", 64)),
        num_points_in_pillar=tuple(config.get("num_points_in_pillar", (4, 4, 4))),
        render_opts=render_opts,
        device=device,
    ).to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    # Train/eval modes
    model.train()
    ema.eval()

    lr = float(config.get("lr", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    bfloat_enable = bool(hasattr(args, "bfloat16") and args.bfloat16)
    scaler = torch.amp.GradScaler(enabled=bfloat_enable)

    # -----------------------------------------------------------------------------
    # Resume checkpoint (TriPlaneCDiT-friendly)
    # -----------------------------------------------------------------------------
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    start_epoch = 0
    train_steps = 0

    resume_path = None
    if os.path.isfile(latest_path):
        resume_path = latest_path
    elif config.get("from_checkpoint", 0):
        resume_path = config.get("from_checkpoint", 0)

    if resume_path:
        logger.info(f"Loading checkpoint from {resume_path}")
        ckp = torch.load(resume_path, map_location="cpu", weights_only=False)

        def _strip_prefix(state_dict):
            out = {}
            for k, v in state_dict.items():
                k = k.replace("_orig_mod.", "")   # torch.compile / DDP sometimes adds this
                k = k.replace("module.", "")      # DDP adds this
                out[k] = v
            return out

        # ---- Load model ----
        if "model" in ckp:
            model_sd = _strip_prefix(ckp["model"])
            missing, unexpected = model.load_state_dict(model_sd, strict=False)
            logger.info(f"Loaded model. missing={len(missing)}, unexpected={len(unexpected)}")
            if len(missing) > 0:
                logger.info(f"  missing keys (show first 20): {missing[:20]}")
            if len(unexpected) > 0:
                logger.info(f"  unexpected keys (show first 20): {unexpected[:20]}")
        else:
            logger.warning("Checkpoint has no 'model' key. Initializing EMA from current model.")
            update_ema(ema, model, decay=0)

        # ---- Load EMA (preferred) ----
        if "ema" in ckp:
            ema_sd = _strip_prefix(ckp["ema"])
            missing, unexpected = ema.load_state_dict(ema_sd, strict=False)
            logger.info(f"Loaded EMA. missing={len(missing)}, unexpected={len(unexpected)}")
            if len(missing) > 0:
                logger.info(f"  missing EMA keys (show first 20): {missing[:20]}")
            if len(unexpected) > 0:
                logger.info(f"  unexpected EMA keys (show first 20): {unexpected[:20]}")
        else:
            logger.warning("Checkpoint has no 'ema' key. Syncing EMA from current model.")
            update_ema(ema, model, decay=0)

        # ---- Load optimizer ----
        if "opt" in ckp:
            try:
                opt.load_state_dict(ckp["opt"])
                logger.info("Loaded optimizer state.")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        # ---- Load epoch/steps/scaler ----
        if "epoch" in ckp:
            start_epoch = int(ckp["epoch"]) + 1
        if "train_steps" in ckp:
            train_steps = int(ckp["train_steps"])

        if bfloat_enable and ("scaler" in ckp):
            try:
                scaler.load_state_dict(ckp["scaler"])
                logger.info("Loaded GradScaler state.")
            except Exception as e:
                logger.warning(f"Failed to load GradScaler state: {e}")

        logger.info(f"Resume done: start_epoch={start_epoch}, train_steps={train_steps}")

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

            dataset = SecondFrameTrainingDataset(
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
                    B, T = x.shape[:2]  # T should be num_cond + 1
                    #two frame all in vae
                    x_flat = x.flatten(0, 1)
                    lat = tokenizer.encode(x_flat).latent_dist.sample().mul_(0.18215)
                    x_lat = lat.unflatten(0, (B, T))  # [B,T,4,Hlat,Wlat]

                # ---- single target frame (gt) ----
                assert T == num_cond + 1, f"Expect T=num_cond+1, but got T={T}, num_cond={num_cond}"
                x_start = x_lat[:, num_cond]  # [B,4,Hlat,Wlat]
                Hlat, Wlat = x_start.shape[-2], x_start.shape[-1]

                # ---- y / rel_t shapes ----
                # y expected: [B,3]
                if y.dim() == 3:         # [B,1,3] -> [B,3]
                    y = y.squeeze(1)
                # rel_t expected: [B]
                if rel_t.dim() == 2 and rel_t.shape[1] == 1:
                    rel_t = rel_t.squeeze(1)

                # ---- cam params shapes ----
                # expected: cam2world [B,4,4], intrinsics [B,3,3]
                # If your dataset accidentally returns [B,1,4,4] or [B,1,3,3], squeeze it.
                if cam2world.dim() == 4 and cam2world.shape[1] == 1:
                    cam2world = cam2world.squeeze(1)
                if intrinsics.dim() == 4 and intrinsics.shape[1] == 1:
                    intrinsics = intrinsics.squeeze(1)

                # ---- diffusion timestep ----
                t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

                # ---- IMPORTANT: if you use TriPlaneCDiT wrapper, do NOT build x_cond here ----
                model_kwargs = dict(
                    y=y,
                    rel_t=rel_t,
                    sat_img=sat_img,          # [B,3,Hs,Ws]
                    cam2world=cam2world,      # [B,4,4]
                    intrinsics=intrinsics,    # [B,3,3]
                )

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
                    "model": model.state_dict(),   # TriPlaneCDiT includes sat2feat + cdit
                    "ema": ema.state_dict(),       # EMA of the whole TriPlaneCDiT
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "train_steps": train_steps,
                    "args": vars(args),
                    # (optional) keep config for reproducibility:
                    "config": config,
                }
                if bfloat_enable:
                    checkpoint["scaler"] = scaler.state_dict()

                ckpt_path = f"{checkpoint_dir}/latest.pth.tar"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

                # (optional) keep periodic numbered snapshots
                if train_steps % (10 * args.ckpt_every) == 0:
                    ckpt_path2 = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                    torch.save(checkpoint, ckpt_path2)
                    logger.info(f"Saved checkpoint to {ckpt_path2}")

            # # Eval
            # if train_steps % args.eval_every == 0 and train_steps > 0:
            #     eval_start = time()
            #     save_dir = os.path.join(experiment_dir, str(train_steps))
            #     sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, config, device, save_dir, bfloat_enable, num_cond)
            #     eval_time = time() - eval_start
            #     logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}s")

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
    parser.add_argument("--gpu", type=int, default=4, help="Which CUDA device index to use")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)