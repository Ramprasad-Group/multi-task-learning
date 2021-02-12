"""Main routine"""

import click
import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from pathlib import Path
import toml
import pandas as pd


def read_config_file():
    config_file = Path('config.toml')
    if not config_file.exists():
        raise UserWarning('Config.toml does not exists.')
    fl = toml.load(config_file.open())
    return fl

@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.version_option(version="0.1.0")
@click.option("--cpu", is_flag=True, help="Use only CPUs")
@click.pass_context
def cli(ctx, cpu):
    """A Multi-Task Toolkit by Christopher Kuenneth @ Georgia Tech in the Ramprasad Research Group"""
    ctx.ensure_object(dict)
    conf = ctx.obj

    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tf.config.set_visible_devices([], "GPU")

    if not cpu:
        # Get all GPUS which are selected in in CUDA_VISIBLE_DEVICES
        gpus = tf.config.experimental.list_physical_devices("GPU")

        # Allow growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # check_config_file()
    config_read = read_config_file()
    conf.update(config_read)

    conf["checkpoint_path"] = Path('./checkpoints')
    conf["checkpoint_path"].mkdir(exist_ok=True)

    conf["results"] = Path('./results')
    conf["results"].mkdir(exist_ok=True)

from mt_pack.tasks.gpr import gpr
cli.add_command(gpr)

from mt_pack.tasks.nn_st import nn_st
cli.add_command(nn_st)

from mt_pack.tasks.nn_mt import nn_mt
cli.add_command(nn_mt)

from mt_pack.tasks.nn_mt2 import nn_mt2
cli.add_command(nn_mt2)


def script():
    cli(obj={})
