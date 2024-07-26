from __future__ import print_function
import argparse
import json
import sys

from utils.eval_squad import evaluate


def main(args):
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../datasets/", help='Dataset file')
    parser.add_argument('--prediction_path', type=str, help='Prediction file')
    parser.add_argument('--evaluation_path', type=str, help='Evaluation output file')

    args = parser.parse_args()
    main(args)