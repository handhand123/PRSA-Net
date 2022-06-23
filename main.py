import os
import time
import warnings

from config.config import get_config
from train import Train
from inference import inference, eval_proposal, eval_detection

warnings.filterwarnings("ignore")


def main(args):
    if args.mode == 'train':
        Train(args)
    elif args.mode == 'infer':
        print("start infer")
        time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("epoch start time:%s" % time1)
        inference(args)
        time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("epoch end time:%s" % time2)

        eval_proposal(args)
        eval_detection(args)



if __name__ == '__main__':
    args = get_config()

    if not os.path.exists(args.output["checkpoint_path"]):
        os.makedirs(args.output['checkpoint_path'])

    if not os.path.exists(args.output["output_path"]):
        os.makedirs(args.output["output_path"])

    main(args)
