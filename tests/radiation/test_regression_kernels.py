#!/usr/bin/env python3
"""Run all four original radiation regressions with production rad_grid methods.

The serial fixture prints the original absolute L1/L2/Linf error measurements;
it does not impose new tolerances or certify the full application's physics.
Distributed boundary actions still require the Octo-TIGER CTest cases. Generate
the required binary Gaussian reference with gen_radiation_reference first.
"""

import argparse
import contextlib
import pathlib
import subprocess
import tempfile


def method(source, signature):
    """Extract a production definition, failing visibly if its interface changes."""
    begin = source.index(signature)
    end = source.index('{', begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def withoutIncludes(path):
    return '\n'.join(line for line in path.read_text().splitlines()
                     if not line.startswith(('#include', '#pragma once')))


def productionSource(root):
    source = (root / 'src/radiation/rad_grid.cpp').read_text()
    signatures = [
        'void rad_grid::allocate(', 'void rad_grid::set_dx(',
        'void rad_grid::set_X(', 'Real rad_grid::maxTimestep(',
        'void rad_grid::compute_flux(', 'void rad_grid::advance(',
        'void rad_grid::sanity_check(', 'void rad_grid::applyRegressionSource(',
        'void rad_grid::set_physical_boundaries(',
        'rad_grid::rad_grid(Real _dx)', 'rad_grid::rad_grid()',
    ]
    support = root / 'tests/radiation'
    return '\n'.join([
        (support / 'regression_fixture.inc').read_text(),
        # octotiger/test_problems/radiation.hpp is now only a forwarding header.
        withoutIncludes(root / 'test_problems/radiation.hpp'),
        withoutIncludes(root / 'src/test_problems/radiation/radiation.cpp'),
        '#define private public',
        withoutIncludes(root / 'octotiger/radiation/rad_grid.hpp'),
        '#undef private',
        *(method(source, signature) for signature in signatures),
        (support / 'regression_checks.inc').read_text(),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--cells', type=int, default=32)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--build-dir', type=pathlib.Path,
                        help='Retain the extracted production fixture and executable here')
    args = parser.parse_args()
    if args.cells < 4 or args.cells > 512 or args.cells % 2:
        parser.error('--cells must be even and between 4 and 512')
    if not args.reference.is_file():
        parser.error(f'reference file does not exist: {args.reference}')
    root = pathlib.Path(__file__).resolve().parents[2]
    build = (contextlib.nullcontext(args.build_dir) if args.build_dir else
             tempfile.TemporaryDirectory(prefix='radiation-regression-'))
    with build as tmp:
        folder = pathlib.Path(tmp)
        folder.mkdir(parents=True, exist_ok=True)
        source, executable = folder / 'test.cpp', folder / 'test'
        source.write_text(productionSource(root))
        flags = (['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
                 if args.sanitize else ['-O3'])
        subprocess.run([args.cxx, '-std=c++23', *flags, '-I' + str(root),
                        '-DTEST_CELLS=' + str(args.cells), str(source),
                        '-o', str(executable)], check=True)
        result = subprocess.run([str(executable), str(args.reference.resolve())],
                                check=True, text=True, stdout=subprocess.PIPE)
        print(result.stdout, end='')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result.stdout)


if __name__ == '__main__':
    main()
