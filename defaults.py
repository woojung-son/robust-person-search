from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.DATASET = "CUHK-SYSU"
_C.INPUT.DATASET_TARGET = ""
_C.INPUT.DATA_ROOT = "data/CUHK-SYSU"
_C.INPUT.DATA_ROOT_TARGET = ""


# TODO: support aspect ratio grouping
# Whether to use aspect ratio grouping for saving GPU memory
# _C.INPUT.ASPECT_RATIO_GROUPING_TRAIN = False

# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 5
_C.INPUT.BATCH_SIZE_TEST = 1

# Number of data loading threads
_C.INPUT.NUM_WORKERS_TRAIN = 8
_C.INPUT.NUM_WORKERS_TEST = 8



_C.SEED = 1
# Directory where output files are written

def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
