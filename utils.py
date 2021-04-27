import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dataset')
    parser.add_argument('-w', '--weight', help='path to model weights')
    parser.add_argument('-c', '--cuda_devices', type=int, help='path to model weights')
    return parser.parse_args()