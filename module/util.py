import os, sys
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.utils import data as torch_data
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__file__)


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
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=0)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def build_solver(cfg, _dataset):
    if cfg.get("split", False):
        splits = _dataset.split(**cfg.split)
    else:
        splits = _dataset.split()
    train_set, valid_set, test_set = splits
    if comm.get_rank() == 0:
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    cfg.task.task = _dataset.target_fields
    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    lr_ratio = cfg.get("lr_ratio", 1.0)
    lr_ratio_2 = cfg.get("lr_ratio_2", 1.0)
    if "model2" in cfg.task:
        if lr_ratio != 1.0 or lr_ratio_2 != 1.0:
            cfg.optimizer.params = [
                {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * lr_ratio},
                {'params': solver.model.model2.parameters(), 'lr': cfg.optimizer.lr * lr_ratio_2},
                {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
            ]
            optimizer = core.Configurable.load_config_dict(cfg.optimizer)
            solver.optimizer = optimizer
    else:
        if lr_ratio != 1.0:
            cfg.optimizer.params = [
                {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * lr_ratio},
                {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
            ]
            optimizer = core.Configurable.load_config_dict(cfg.optimizer)
            solver.optimizer = optimizer

    fix_encoder = cfg.get("fix_encoder", False)
    if fix_encoder:
        for p in task.model.parameters():
            p.requires_grad = False
    fix_encoder_2 = cfg.get("fix_encoder_2", False)
    if fix_encoder_2:
        for p in task.model2.parameters():
            p.requires_grad = False

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint, load_optimizer=False)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load protein checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        load_task = cfg.get("load_task", False)
        if load_task:
            result = task.load_state_dict(model_dict["model"])
        else:
            load_unsupervised = cfg.get("load_unsupervised", False)
            if load_unsupervised:
                result = task.model.model.load_state_dict(model_dict, strict=False)
            else:
                result = task.model.load_state_dict(model_dict, strict=False)

        if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
            print("Model parameters loaded successfully")
        else:
            print(f"Missing keys: {result.missing_keys}")
            print(f"Unexpected keys: {result.unexpected_keys}")

    if cfg.get("model_checkpoint_2") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load protein checkpoint from %s" % cfg.model_checkpoint_2)
        cfg.model_checkpoint_2 = os.path.expanduser(cfg.model_checkpoint_2)
        model_dict = torch.load(cfg.model_checkpoint_2, map_location=torch.device('cpu'))
        result = task.model2.load_state_dict(model_dict, strict=False)

        if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
            print("Model parameters loaded successfully")
        else:
            print(f"Missing keys: {result.missing_keys}")
            print(f"Unexpected keys: {result.unexpected_keys}")

    return solver, scheduler


def build_downstream_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if cfg.task['class'] == 'MultipleBinaryClassification':
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    else:
        cfg.task.task = dataset.tasks
    task = core.Configurable.load_config_dict(cfg.task)

    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        assert cfg.task.model["class"] == "FusionNetwork"
        cfg.optimizer.params = [
            {'params': solver.model.model.sequence_model.parameters(),
             'lr': cfg.optimizer.lr * cfg.sequence_model_lr_ratio},
            {'params': solver.model.model.structure_model.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        task.model.load_state_dict(model_dict)

    return solver, scheduler


def build_pretrain_solver(cfg, dataset):
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#dataset: %d" % (len(dataset)))

    task = core.Configurable.load_config_dict(cfg.task)
    if "fix_sequence_model" in cfg:
        if cfg.task["class"] == "Unsupervised":
            model_dict = cfg.task.model.model
        else:
            model_dict = cfg.task.model
        assert model_dict["class"] == "FusionNetwork"
        for p in task.model.model.sequence_model.parameters():
            p.requires_grad = False
    cfg.optimizer.params = [p for p in task.parameters() if p.requires_grad]
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, dataset, None, None, optimizer, **cfg.engine)

    return solver