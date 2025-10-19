import argparse
import os.path as osp
import os

from datasets.build import build_test_loader
import multiprocessing as mp
from defaults import get_default_cfg
def main(args):
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print('mp.set_start_method should be called once')
        pass

    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.freeze()
    
    if args.data_root:
        cfg['INPUT']['DATA_ROOT'] = args.data_root
    if cfg.INPUT.DATASET == 'CUHK-SYSU':
        save_dir = osp.join(os.path.dirname(os.path.abspath(cfg['INPUT']['DATA_ROOT'])), 'CUHK-SYSU-C', 'Image', 'SSM')
    elif cfg.INPUT.DATASET == 'PRW':
        save_dir = osp.join(os.path.dirname(os.path.abspath(cfg['INPUT']['DATA_ROOT'])), 'PRW-C', 'frames')
    print(f'save_dir: {save_dir}')
    save_dir = osp.join('/mnt/nas4/woojung/datasets_debug', f'PRW-C_test2', 'frames')

    os.makedirs(save_dir, exist_ok=True)
    gallery_loader, _ = build_test_loader(cfg)
    for i, (images, targets) in enumerate(gallery_loader):
        save_path = osp.join(save_dir, f"{targets[0]['img_name']}")
        images[0].save(save_path)
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument("--data_root", required=False, default='', type=str, help="data root directory")
    args = parser.parse_args()
    main(args)
