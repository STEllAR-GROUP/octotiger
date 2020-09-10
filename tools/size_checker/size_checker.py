#!/usr/bin/env python3

import argparse
from os import listdir
from os import path


def fail_if_file_sizes_differ(original, target):
    assert path.exists(original), '"' + original + \
        '" does not exist. Check failed.'
    assert path.isfile(original), '"' + original + \
        '" is not a file. Check failed.'

    assert path.exists(target), target + ' does not exist. Check failed.'
    assert path.isfile(target), target + ' is not a file. Check failed.'

    assert path.getsize(target) == path.getsize(original), \
        'File sizes differ between "{}" and "{}". Check failed.'.format(target, original)


def main():
    parser = argparse.ArgumentParser(
        description='Check generated Silo file size(s) against original')
    parser.add_argument('original',
                        help='Original Silo file to compare against')
    parser.add_argument('target',
                        help='Generated Silo file')

    args = parser.parse_args()

    # check original file sizes
    fail_if_file_sizes_differ(args.original, args.target)

    # check for sub-silo files
    ofn = path.basename(args.original) + '.data'
    ofp = path.join(path.dirname(args.original), ofn)

    tfn = path.basename(args.target) + '.data'
    tfp = path.join(path.dirname(args.target), ofn)
    if path.isdir(ofp):
        assert path.isdir(
            tfp), 'Target silo data file does not exist. Check failed.'

        ossf = [path.join(ofp, f) for f in listdir(ofp) if path.isfile(
            path.join(ofp, f)) and path.splitext(path.join(ofp, f))[1] == '.silo']
        tssf = [path.join(tfp, f) for f in listdir(tfp) if path.isfile(
            path.join(tfp, f)) and path.splitext(path.join(tfp, f))[1] == '.silo']

        assert len(ossf) == len(
            tssf), 'Target silo data has a different number of Silo files. Check failed.'

        for o, t in zip(ossf, tssf):
            fail_if_file_sizes_differ(o, t)


if __name__ == '__main__':
    main()
