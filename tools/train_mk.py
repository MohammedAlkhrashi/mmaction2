# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from distutils.command.config import config
import os
import os.path as osp
import time
import warnings
from mmaction.datasets.audio_visual_dataset_mk import AudioVisualDatasetMk

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.core.optimizer.combined_optimizer_constructor import CombinedOptimizerConstructor
from mmaction.datasets import build_dataset
from mmaction.datasets.pipelines.ensuring import EnsureFixedKeys
from mmaction.models import build_model
from mmaction.models.recognizers.audio_video_recognizer import VideoAudioRecognizer
from mmaction.models.recognizers.base import BaseRecognizer
from mmaction.utils import (
    collect_env,
    get_root_logger,
    register_module_hooks,
    setup_multi_processes,
)


from mmaction.datasets.audio_visual_dataset import AudioVisualDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a recognizer")
    parser.add_argument("config_video", help="train config file path")
    parser.add_argument("config_audio", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--test-last",
        action="store_true",
        help="whether to test the checkpoint after training",
    )
    parser.add_argument(
        "--test-best",
        action="store_true",
        help=("whether to test the best checkpoint (if applicable) after " "training"),
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff-seed",
        action="store_true",
        help="Whether or not set different seeds for different ranks",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. For example, "
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def settings_from_config(args, cfg, config_path):
    cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(config_path))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None or args.gpus is not None:
        warnings.warn(
            "The Args `gpu_ids` and `gpus` are only used in non-distributed "
            "mode and we highly encourage you to use distributed mode, i.e., "
            "launch training with dist_train.sh. The two args will be "
            "deperacted."
        )
        if args.gpu_ids is not None:
            warnings.warn(
                "Non-distributed training can only use 1 gpu now. We will "
                "use the 1st one in gpu_ids. "
            )
            cfg.gpu_ids = [args.gpu_ids[0]]
        elif args.gpus is not None:
            warnings.warn("Non-distributed training can only use 1 gpu now. ")
            cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault("omnisource", False)

    # The flag is used to register module's hooks
    cfg.setdefault("module_hooks", [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config_path)))
    # init logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config: {cfg.pretty_text}")

    # set random seeds
    seed = init_random_seed(args.seed, distributed=distributed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta["seed"] = seed
    meta["config_name"] = osp.basename(config_path)
    meta["work_dir"] = osp.basename(cfg.work_dir.rstrip("/\\"))

    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    # if cfg.omnisource:
    #     # If omnisource flag is set, cfg.data.train should be a list
    #     assert isinstance(cfg.data.train, list)
    #     datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    # else:
        # datasets = [build_dataset(cfg.data.train)]

    # if len(cfg.workflow) == 2:
    #     # For simplicity, omnisource is not compatible with val workflow,
    #     # we recommend you to use `--validate`
    #     assert not cfg.omnisource
    #     if args.validate:
    #         warnings.warn('val workflow is duplicated with `--validate`, '
    #                       'it is recommended to use `--validate`. see '
    #                       'https://github.com/open-mmlab/mmaction2/pull/123')
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    return distributed, timestamp, meta, model, test_option


def build_dataset_mk(cfg_video, cfg_audio):
    def combine_pipelines(video_pipleline, audio_pipleline):
        initial_keys = [
            # "total_frames",
            "audio_path",
            "filename",
            "frame_dir",
            "offset",
            "label",
            "start_index",
            "modality"
        ] # Initial keys that are used to initiate the pipeline. 
        final_pipeline = (
            [EnsureFixedKeys(keys=initial_keys)]
            + audio_pipleline[:-2]
            + [EnsureFixedKeys(keys=["audios"])] 
            + video_pipleline[:-2]
        )
        final_pipeline = final_pipeline + [
            dict(type="Collect", keys=["imgs", "audios", "label"], meta_keys=[]),
            dict(type="ToTensor", keys=["imgs", "audios", "label"]),
        ]
        
        return final_pipeline


    video_prefix = cfg_video.data_root
    audio_prefix = cfg_audio.data_root

    train_ann_file = cfg_video.ann_file_train
    train_set = AudioVisualDatasetMk(
        ann_file=train_ann_file,
        video_prefix=video_prefix,
        audio_prefix=audio_prefix,
        pipeline=combine_pipelines(cfg_video.train_pipeline, cfg_audio.train_pipeline),
    )

    val_ann_file = cfg_video.ann_file_val
    val_set = AudioVisualDatasetMk(
        ann_file=val_ann_file,
        video_prefix=video_prefix,
        audio_prefix=audio_prefix,
        pipeline=combine_pipelines(cfg_video.val_pipeline, cfg_audio.val_pipeline),
    )

    return train_set,val_set

def build_model_mk(model_video, model_audio):
    return VideoAudioRecognizer(model_video, model_audio)

def combine_optimizers_config(cfg_video,cfg_audio):
    combined_full_config = cfg_video.copy() # video is default.

    combined_optim_config = dict()
    combined_optim_config['video'] = cfg_video.optimizer.copy()
    combined_optim_config['video_module_name'] = "model_video"

    combined_optim_config['audio'] = cfg_audio.optimizer.copy()
    combined_optim_config['audio_module_name'] = "model_audio"


    combined_optim_config['head'] = cfg_video.optimizer.copy() # For now using same optim as video for head. 
    combined_optim_config['head_module_name'] = "av_head"


    combined_optim_config['constructor'] = "CombinedOptimizerConstructor"

    v_optim_type = combined_optim_config['video'].pop("type")
    a_optim_type = combined_optim_config['audio'].pop("type")
    _ = combined_optim_config['head'].pop("type")
    
    assert v_optim_type == a_optim_type
    final_optim_type = v_optim_type # same as a_optim_type

    combined_optim_config['type'] = final_optim_type
    combined_full_config['optimizer'] = combined_optim_config
    
    return combined_full_config

def final_settins_from_configs(args, cfg_video, cfg_audio):
    train_dataset_av, val_dataset_av= build_dataset_mk(cfg_video, cfg_audio)
    train_dataset_av = [train_dataset_av]

    (
        distributed_video,
        timestamp_video,
        meta_video,
        model_video,
        test_option_video,
    ) = settings_from_config(args, cfg_video, args.config_video)
    # _, _, _, model_audio, _ = settings_from_config(args, cfg_audio, args.config_audio)


    cfg_audio.setdefault("module_hooks", [])
    model_audio = build_model(
        cfg_audio.model, train_cfg=cfg_audio.get("train_cfg"), test_cfg=cfg_audio.get("test_cfg")
    )
    if len(cfg_audio.module_hooks) > 0:
        register_module_hooks(model_audio, cfg_audio.module_hooks)




    config_final =  combine_optimizers_config(cfg_video,cfg_audio)
    model_av = build_model_mk(model_video, model_audio)
    return (
        distributed_video,
        timestamp_video,
        meta_video,
        model_av,
        train_dataset_av,
        val_dataset_av,
        test_option_video,
        config_final,
        
    )


def main():
    args = parse_args()

    cfg_video = Config.fromfile(args.config_video)
    cfg_audio = Config.fromfile(args.config_audio)
    (
        distributed,
        timestamp,
        meta,
        model,
        train_datasets,
        val_dataset,
        test_option,
        cfg,
    ) = final_settins_from_configs(args, cfg_video, cfg_audio)

    train_model(
        model,
        train_datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta,
        val_dataset=val_dataset
    )


if __name__ == "__main__":
    main()
