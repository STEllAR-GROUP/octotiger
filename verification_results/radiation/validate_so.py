#!/usr/bin/env python3
"""Compile production M1/rad_grid/driver methods with serial infrastructure fixtures.
No HPX, MPI, Silo, or FFTW required. This does not replace a distributed run.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


def method(text, signature):
    start = text.index(signature)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def productionSource(root):
    support = root/'verification_results/radiation/tests_so'
    source = (root/'src/radiation/rad_grid.cpp').read_text()
    header = '\n'.join(line for line in (root/'octotiger/radiation/rad_grid.hpp').read_text().splitlines()
                       if not line.startswith('#include'))
    signatures = ['Real radiationGasInternal(', 'void rad_grid::allocate(',
                  'void rad_grid::set_dx(', 'void rad_grid::set_X(',
                  'Real rad_grid::maxTimestep(', 'void rad_grid::compute_flux(',
                  'void rad_grid::advance(', 'void rad_grid::sanity_check(',
                  'void rad_grid::applyRegressionSource(',
                  'radiationConservation::Totals rad_grid::takeConservation(',
                  'void rad_grid::accountBoundaryFlux(', 'void rad_grid::set_physical_boundaries(',
                  'void rad_grid::compute_mmw(', 'void rad_grid::rad_imp(',
                  'Real rad_grid::hydroSignalSpeed(',
                  'void rad_grid::prepareSources(', 'void rad_grid::finishSources(',
                  'void rad_grid::set_flux_restrict(', 'std::vector<Real> rad_grid::get_flux_restrict(',
                  'rad_grid::rad_grid(Real _dx)', 'rad_grid::rad_grid()']
    cpp = (support/'fixture.inc').read_text() + '\n' + header
    opacity = (root/'octotiger/radiation/opacities.hpp').read_text()
    cpp += '\n' + method(opacity, 'inline Real radiationAbsorption(')
    cpp += '\n' + method(opacity, 'inline Real radiationTransport(')
    cpp += '\n#undef private\n' + '\n'.join(method(source, signature) for signature in signatures)
    cpp += '\n' + (support/'driver.inc').read_text()
    cpp += '\n' + method(source, 'void node_server::all_rad_bounds(')
    cpp += '\n' + method(source, 'void node_server::compute_radiation(')
    return cpp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--opacity-checks', action='store_true', help='Also run grey opacity production-source tests')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    support = root/'verification_results/radiation/tests_so'
    with tempfile.TemporaryDirectory(prefix='octotiger-so-') as directory:
        path = Path(directory)
        cpp = productionSource(root)
        checks = (support/'checks.inc').read_text()
        if args.opacity_checks:
            checks = checks.replace('int main() {', (support/'opacity_checks.inc').read_text() + '\nint main() {')
            checks = checks.replace('kernelChecks();thermalChecks();', 'opacitySourceChecks();kernelChecks();thermalChecks();')
        cpp += '\n' + checks
        (path/'check.cpp').write_text(cpp)
        flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
        if args.release:
            flags += ['-DNDEBUG']
        subprocess.run([args.cxx, '-std=c++23', '-Wall', '-Wextra', '-Wno-unused-parameter',
                        *flags, '-I'+str(root), str(path/'check.cpp'), '-o', str(path/'check')], check=True)
        subprocess.run([str(path/'check')], check=True)


if __name__ == '__main__':
    main()
