import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/thumos_i3d_PRSA.yaml', nargs='?')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        tmp = f.read()
        data = yaml.load(tmp, Loader=yaml.FullLoader)

    for k, v in data.items():
        setattr(args, k, v)

    return args
