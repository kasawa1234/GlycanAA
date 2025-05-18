import os
import sys
import pprint

import torch

from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module import custom_data, custom_datasets, custom_models, custom_tasks, util


def save(solver, path):
    if isinstance(solver.model, tasks.Unsupervised):
        model = solver.model.model.model
    else:
        model = solver.model.model

    if comm.get_rank() == 0:
        logger.warning("Save checkpoint to %s" % path)
    path = os.path.expanduser(path)
    if comm.get_rank() == 0:
        torch.save(model.state_dict(), path)
    comm.synchronize()


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_pretrain_solver(cfg, _dataset)

    step = cfg.get("save_interval", 1)
    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)

        solver.train(**kwargs)

        save(solver, cfg.save_path + "model_epoch_%d.pth" % (i + kwargs["num_epoch"]))
