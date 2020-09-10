#!/usr/bin/env python3

import argparse
import sys
from os import listdir
from os import path


def fail_if_not(expected, message):
    if not expected:
        print(message, file=sys.stderr)
        sys.exit(1)


def fail_if_file_sizes_differ(original, target):
    fail_if_not(path.exists(original), '"' + original +
                '" does not exist.')
    fail_if_not(path.isfile(original), '"' + original +
                '" is not a file.')

    fail_if_not(path.exists(target), '"{}" does not exist.'.format(target))
    fail_if_not(path.isfile(target), '"{}" is not a file.'.format(target))

    fail_if_not(path.getsize(target) == path.getsize(original),
                'File sizes differ between "{}" and "{}".'.format(target, original))


def main():
    parser = argparse.ArgumentParser(
        description='Check generated Silo file size(s) against original')
    parser.add_argument(
        'original', help='Original Silo file to compare against')
    parser.add_argument('target', help='Generated Silo file')

    args = parser.parse_args()

    # check original file sizes
    fail_if_file_sizes_differ(args.original, args.target)

    # check for sub-silo files
    ofn = path.basename(args.original) + '.data'
    ofp = path.join(path.dirname(args.original), ofn)

    tfn = path.basename(args.target) + '.data'
    tfp = path.join(path.dirname(args.target), tfn)
    if path.isdir(ofp):
        fail_if_not(path.isdir(tfp),
                    'Target silo data directory does not exist.')

        ossf = [path.join(ofp, f) for f in listdir(ofp) if path.isfile(
            path.join(ofp, f)) and path.splitext(path.join(ofp, f))[1] == '.silo']
        tssf = [path.join(tfp, f) for f in listdir(tfp) if path.isfile(
            path.join(tfp, f)) and path.splitext(path.join(tfp, f))[1] == '.silo']

        fail_if_not(len(ossf) == len(tssf),
                    'Target silo data has a different number of Silo files.')

        for o, t in zip(ossf, tssf):
            fail_if_file_sizes_differ(o, t)


if __name__ == '__main__':
    main()
