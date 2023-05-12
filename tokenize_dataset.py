import argparse
import json
import os

import tqdm.auto as tqdm

import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(pat