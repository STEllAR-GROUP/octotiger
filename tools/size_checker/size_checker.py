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


def get_data_dirname(p):
    fn = path.basename(p) + '.data'
    return path.join(path.dirname(p), fn)


def listdir_silo(fp):
    return [path.join(fp, f) for f in listdir(fp) if path.isfile(
            path.join(fp, f)) and path.splitext(path.join(fp, f))[1] == '.silo']


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
    ofp = get_data_dirname(args.original)
    tfp = get_data_dirname(args.target)

    if path.isdir(ofp):
        fail_if_not(path.isdir(tfp),
                    'Target silo data directory does not exist.')

        ossf = listdir_silo(ofp)
        tssf = listdir_silo(tfp)

        fail_if_not(len(ossf) == len(
            tssf), 'Silo file count differs: {} != {}'.format(len(ossf), len(tssf)))

        for o, t in zip(ossf, tssf):
            fail_if_file_sizes_differ(o, t)


if __name__ == '__main__':
    main()
