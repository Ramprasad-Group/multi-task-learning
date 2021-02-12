import tensorflow as tf
from pathlib import Path
import shutil

def get_checkpoint_log(conf, name):
    name = str(name)
    ckpt_path = str(conf["checkpoint_path"].joinpath(name, "single.checkpoint"))
    logdir = f"./logs/{name}/"

    Path(logdir).exists() and shutil.rmtree(logdir)
    return ckpt_path, logdir
