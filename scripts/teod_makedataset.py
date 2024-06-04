#!/usr/bin/env python3
import argparse


def ConfigureArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--lmdb_dir', type=str, required=True)
    parser.add_argument('--view4label_dir', type=str, required=True)