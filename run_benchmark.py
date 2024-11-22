import os
import hydra
import sys
import math
import pprint
import shutil
import logging
import argparse
import numpy as np
import time
import yaml
import easydict
from pathlib import Path
import uuid

import torch
from torch import distributed as dist
from omegaconf import OmegaConf, DictConfig

import torchdrug
from torchdrug import core, datasets, tasks, models, layers
from torchdrug.utils import comm

from plaid.benchmarking import ours, flip


def resolve_cfg(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = easydict.EasyDict(cfg)
    return cfg


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    hashid = uuid.uuid4().hex[:7]
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    output_dir = os.path.join(
        os.path.expanduser(cfg.output_dir),
        cfg.task["class"],
        cfg.dataset["class"],
        cfg.task.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S") + "_" + hashid,
    )

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)

    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        try:
            os.remove(file_name)
        except:
            pass

    os.chdir(output_dir)
    return output_dir


def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def build_solver(cfg, logger):
    # build dataset
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        train_set, valid_set, test_set = _dataset.split(["train", "valid", cfg.test_split])
    else:
        train_set, valid_set, test_set = _dataset.split()
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    # build task model
    if cfg.task["class"] in ["PropertyPrediction", "InteractionPrediction"]:
        cfg.task.task = _dataset.tasks
    task = core.Configurable.load_config_dict(cfg.task)

    # fix the pre-trained encoder if specified
    # fix_encoder = cfg.get("fix_encoder", False)
    # fix_encoder2 = cfg.get("fix_encoder2", False)
    # if fix_encoder:
    #     for p in task.model.parameters():
    #         p.requires_grad = False
    # if fix_encoder2:
    #     for p in task.model2.parameters():
    #         p.requires_grad = False

    # build solver
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if not "scheduler" in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)
    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {"params": solver.model.model.parameters(), "lr": cfg.optimizer.lr * cfg.lr_ratio},
            {"params": solver.model.mlp.parameters(), "lr": cfg.optimizer.lr},
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint, load_optimizer=False)

    return solver


def train_and_validate(cfg, solver):
    step = math.ceil(cfg.train.num_epoch / 10)
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        # solver.save("model_epoch_%d.pth" % solver.epoch)
        if "test_batch_size" in cfg:
            solver.batch_size = cfg.test_batch_size
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        solver.batch_size = cfg.engine.batch_size

        score = []
        for k, v in metric.items():
            if k.startswith(cfg.eval_metric):
                if "root mean squared error" in cfg.eval_metric:
                    score.append(-v)
                else:
                    score.append(v)
        score = sum(score) / len(score)
        if score > best_score:
            best_score = score
            best_epoch = solver.epoch

    # solver.load("model_epoch_%d.pth" % best_epoch)
    # return solver, best_epoch


def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
    solver.model.split = "valid"
    solver.evaluate("valid")
    solver.model.split = "test"
    solver.evaluate("test")

    return


@hydra.main(version_base=None, config_path="configs/benchmark", config_name="beta")
def main(cfg: DictConfig) -> None:
    cfg = resolve_cfg(cfg)
    set_seed(0)  # TODO: run with more seeds

    output_dir = create_working_directory(cfg)
    logger = get_root_logger()
    os.chdir(output_dir)

    solver = build_solver(cfg, logger)
    train_and_validate(cfg, solver)
    # if comm.get_rank() == 0:
    #     logger.warning("Best epoch on valid: %d" % best_epoch)
    # test(cfg, solver)


if __name__ == "__main__":
    main()
