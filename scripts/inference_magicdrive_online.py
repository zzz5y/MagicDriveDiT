import os
import gc
import sys
import time
import copy
from pprint import pformat
from datetime import timedelta
from functools import partial

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
if not torch.cuda.is_available() or DEVICE_TYPE == 'npu':
    USE_NPU = True
    os.environ['DEVICE_TYPE'] = "npu"
    DEVICE_TYPE = "npu"
    print("Enable NPU!")
    try:
        # just before torch_npu, let xformers know there is no gpu
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import magicdrivedit.utils.module_contrib

import colossalai
import torch.distributed as dist
from torch.utils.data import Subset
from einops import rearrange, repeat
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from mmengine.runner import set_random_seed
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from mmcv.parallel import DataContainer

from magicdrivedit.acceleration.parallel_states import (
    set_sequence_parallel_group,
    get_sequence_parallel_group,
    set_data_parallel_group,
    get_data_parallel_group
)
from magicdrivedit.datasets import save_sample,save_sample_inf
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.models.text_encoder.t5 import text_preprocessing
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.config_utils import parse_configs, define_experiment_workspace, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
from magicdrivedit.utils.inference_utils import (
    apply_mask_strategy,
    get_save_path_name,
    concat_6_views_pt,
    add_null_condition,
    enable_offload,
)
from magicdrivedit.utils.misc import (
    reset_logger,
    is_distributed,
    is_main_process,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent,
)
from magicdrivedit.utils.train_utils import sp_vae


TILING_PARAM = {
    "default": dict(),  # it is designed for CogVideoX's 720x480, 4.5 GB
    "384": dict(  # about 14.2 GB
        tile_sample_min_height = 384,  # should be 48n
        tile_sample_min_width = 720,  # should be 40n
    ),
}

def load_data_from_pt(file_path):
    """
    从 .pt 文件中加载数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return torch.load(file_path)


def set_omegaconf_key_value(cfg, key, value):
    p, m = key.rsplit(".", 1)
    node = cfg
    for pk in p.split("."):
        node = getattr(node, pk)
    node[m] = value

    
def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    if cfg.get("vsdebug", False):
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    # == dataset config ==
    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets = []
        val_datasets = []
        for (res, data_cfg_name), overrides in zip(
                cfg.data_cfg_names, cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)):
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:
        dataset, val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)
    if cfg.get("use_train", False):
        cfg.dataset = dataset
        tag = cfg.get("tag", "")
        cfg.tag = "train" if tag == "" else f"{tag}_train"
    else:
        cfg.dataset = val_dataset
    # set img_collate_param
    if hasattr(cfg.dataset, "img_collate_param"):
        cfg.dataset.img_collate_param.is_train = False  # Important!
    else:
        for d in cfg.dataset.cfg:
            d[1].img_collate_param.is_train = False  # Important!
    cfg.batch_size = 1
    # for lower cpu memory in dataloading
    cfg.ignore_ori_imgs = cfg.get("ignore_ori_imgs", True)
    if cfg.ignore_ori_imgs:
        cfg.dataset.drop_ori_imgs = True

    # for lower gpu memory in vae decoding
    cfg.vae_tiling = cfg.get("vae_tiling", None)

    # edit annotations
    if cfg.get("allow_class", None) != None:
        cfg.dataset.allow_class = cfg.allow_class
    if cfg.get("del_box_ratio", None) != None:
        cfg.dataset.del_box_ratio = cfg.del_box_ratio
    if cfg.get("drop_nearest_car", None) != None:
        cfg.dataset.drop_nearest_car = cfg.drop_nearest_car

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if USE_NPU:  # disable some kernels
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # == init distributed env ==
    if is_distributed():
        # colossalai.launch_from_torch({})
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        cfg.sp_size = dist.get_world_size()
    else:
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0,
            init_method="tcp://localhost:12355")
        cfg.sp_size = 1
    coordinator = DistCoordinator()
    if cfg.sp_size > 1:
        DP_AXIS, SP_AXIS = 0, 1
        dp_size = dist.get_world_size() // cfg.sp_size
        pg_mesh = ProcessGroupMesh(dp_size, cfg.sp_size)
        dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
        sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
        set_sequence_parallel_group(sp_group)
        print(f"Using sp_size={cfg.sp_size}")
    else:
        # TODO: sequence_parallel_group unset!
        dp_group = dist.group.WORLD
    set_data_parallel_group(dp_group)
    enable_sequence_parallelism = cfg.sp_size > 1
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init exp_dir ==
    cfg.outputs = cfg.get("outputs", "outputs/test")
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    cfg.save_dir = os.path.join(exp_dir, "generation")
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger ==
    logger = reset_logger(exp_dir)
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    if cfg.get("val", None):
        validation_index = cfg.val.validation_index
        if validation_index == "all":
            raise NotImplementedError()
        cfg.num_sample = cfg.val.get("num_sample", 1)
        cfg.scheduler = cfg.val.get("scheduler", cfg.scheduler)
    else:
        validation_index = cfg.get("validation_index", "all")

    # == build dataset ==
    logger.info("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    if validation_index == "even":
        idxs = list(range(0, len(dataset), 2))
        dataset = torch.utils.data.Subset(dataset, idxs)
    elif validation_index == "odd":
        idxs = list(reversed(list(range(1, len(dataset), 2))))  # reversed!
        dataset = torch.utils.data.Subset(dataset, idxs)
    elif validation_index != "all":
        dataset = torch.utils.data.Subset(dataset, validation_index)
    logger.info(f"Your validation index: {validation_index}")
    logger.info("Dataset contains %s samples.", len(dataset))
    logger.info(f"{dataset=}")
    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=isinstance(validation_index, str),  # changed
        drop_last=False,  # changed
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    # NOTE: set to true/false,
    # https://github.com/huggingface/transformers/issues/5486
    # if the program gets stuck, try set it to false
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    if cfg.vae_tiling:
        vae.module.enable_tiling(**TILING_PARAM[str(cfg.vae_tiling)])
        logger.info(f"VAE Tiling is enabled with {TILING_PARAM[str(cfg.vae_tiling)]}")

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=(None, None, None),
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    cfg.cpu_offload = cfg.get("cpu_offload", False)
    if cfg.cpu_offload:
        text_encoder.t5.model.to("cpu")
        model.to("cpu")
        vae.to("cpu")
        text_encoder.t5.model, model, vae, last_hook = enable_offload(
            text_encoder.t5.model, model, vae, device)
    # == load prompts ==
    # prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)

    # == prepare arguments ==
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # == Iter over all samples ==
    start_step = 0
    assert batch_size == 1
    sampler.set_epoch(0)
    dataloader_iter = iter(dataloader)
    
    pt_file_path = "./pt/inference_data.pt"
    while True:
        
        # == Iter over number of sampling for one prompt ==

        # 加载 .pt 数据
        data = load_data_from_pt(pt_file_path)
        video_clips = []
        z, masks, batch_prompts, neg_prompts, _model_args,B,T,NC = (
            data["z"],
            data["masks"],
            data["batch_prompts"],
            data["neg_prompts"],
            data["_model_args"],
            data["B"],
            data["T"],
            data["NC"],
        )
        batch_prompts[0]="A driving scene image at singapore-queenstown.Night Parked cars, peds on sidewalk, peds, trees, intersection"
        neg_prompts = ["Daytime. rain, boston-seaport"]
        y = batch_prompts
        # ms = mask_strategy[i : i + batch_size]
        ms = [""] * len(y)
        # refs = reference_path[i : i + batch_size]
        refs = [""] * len(y)
        
        start_idx = 0
        save_fps = int(_model_args['fps'][0])
        for ns in range(num_sample):
            gc.collect()
            torch.cuda.empty_cache()
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=y[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=ns,
                )
                for idx in range(len(y))
            ]

            # == inference ==
            masks = None
            masks = apply_mask_strategy(z, refs, ms, 0, align=None)
            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                neg_prompts=neg_prompts,
                device=device,
                additional_args=_model_args,
                progress=verbose >= 1,
                mask=masks,
            )
            samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
            if cfg.sp_size > 1:
                samples = sp_vae(
                    samples.to(dtype),
                    partial(vae.decode, num_frames=_model_args["num_frames"]),
                    get_sequence_parallel_group(),
                )
            else:
                samples = vae.decode(samples.to(dtype), num_frames=_model_args["num_frames"])
            samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
            if cfg.cpu_offload:
                last_hook.offload()
            if is_main_process():
                vid_samples = []
                for sample in samples:
                    vid_samples.append(
                        concat_6_views_pt(sample, oneline=False)
                    )
                samples = torch.stack(vid_samples, dim=0)
                video_clips.append(samples)
                del vid_samples
            del samples
            coordinator.block_all()

            # == save samples ==
            torch.cuda.empty_cache()
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 1:
                        logger.info(f"Prompt: {batch_prompt}")
                        if neg_prompts is not None:
                            logger.info(f"Neg-prompt: {neg_prompts[idx]}")
                    save_path = save_paths[idx]
                    video = [video_clips[0][idx]]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample_inf(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        high_quality=True,
                        verbose=verbose >= 2,
                        save_per_n_frame=cfg.get("save_per_n_frame", -1),
                        force_image=cfg.get("force_image", False),
                    )
            del video_clips
            coordinator.block_all()
        
        # save_gt
        if is_main_process() and not cfg.ignore_ori_imgs:
            torch.cuda.empty_cache()
            samples = rearrange(x, "(B NC) C T H W -> B NC C T H W", NC=NC)
            for idx, sample in enumerate(samples):
                vid_sample = concat_6_views_pt(sample, oneline=False)
                save_path = save_sample(
                    vid_sample,
                    fps=save_fps,
                    save_path=os.path.join(save_dir, f"gt_{start_idx + idx:04d}"),
                    high_quality=True,
                    verbose=verbose >= 2,
                    save_per_n_frame=cfg.get("save_per_n_frame", -1),
                    force_image=cfg.get("force_image", False),
                )
            del samples, vid_sample
        coordinator.block_all()
        start_idx += len(batch_prompts)
    

    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx - cfg.get("start_index", 0), save_dir)
    coordinator.destroy()


if __name__ == "__main__":
    main()
