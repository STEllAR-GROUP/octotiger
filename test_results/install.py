#!/usr/bin/env python3
"""Install a small, context-checked export hook in the updated radiation tests."""
import argparse
import difflib
from pathlib import Path
import shutil
import sys

INCLUDES = '#include "octotiger/test_problems/radiation.hpp"\n#include "octotiger/test_problems/radiation/plot_output.hpp"\n'
START = '''
	// RADIATION_PLOT_EXPORT_BEGIN: opt in by creating datadir/radiation-slices.
	radiationTests::SliceOutput radiationSlice(radiationRegressionProblem(),
		opts().data_dir, double(t), double(dx), 2 * double(opts().xscale));'''
CAPTURE = '''				radiationSlice.capture(double(X[XDIM][iii]), double(X[YDIM][iii]),
					double(X[ZDIM][iii]), A, opts().n_fields, [&](int f) {
						return double(rad_grid_ptr->get_field(f, i - H_BW + R_BW,
							j - H_BW + R_BW, k - H_BW + R_BW));
					});
'''

def patched(text):
    if 'RADIATION_PLOT_EXPORT_BEGIN' in text:
        if 'radiationSlice.capture(' not in text or 'radiationSlice.finish();' not in text:
            raise ValueError('Incomplete existing plot hook; inspect src/grid.cpp')
        return text
    begin = text.index('analytic_t grid::compute_analytic(Real t) {')
    end = text.index('\nvoid grid::allocate()', begin)
    old = text[begin:end]
    anchor = '\tconst Real dv = dx * dx * dx;'
    loop = '\t\t\t\tfor (integer field = 0; field != opts().n_fields; ++field) {'
    if any(old.count(s) != 1 for s in (anchor, loop, '\treturn a;')):
        raise ValueError('compute_analytic differs from the tested version; no files changed')
    new = old.replace(anchor, anchor + START).replace(loop, CAPTURE + loop)
    new = new.replace('\treturn a;', '\tradiationSlice.finish();\n\treturn a;')
    return INCLUDES + text[:begin] + new + text[end:]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parent.parent)
    p.add_argument('--check', action='store_true', help='print the patch without changing files')
    a = p.parse_args(); root = a.root.resolve()
    if not (root/'octotiger/test_problems/radiation/reference.hpp').is_file():
        raise ValueError('Install the new radiation CTest/reference update first')
    source = root/'src/grid.cpp'; old = source.read_text(); new = patched(old)
    if a.check:
        print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
              fromfile='a/src/grid.cpp', tofile='b/src/grid.cpp')), end='')
        return
    header = root/'octotiger/test_problems/radiation/plot_output.hpp'
    supplied = Path(__file__).resolve().parent/'support/plot_output.hpp'
    if header.exists() and header.read_bytes() != supplied.read_bytes():
        raise ValueError(f'{header} already contains different code; no files changed')
    if new != old:
        backup = source.with_name('grid.cpp.before-radiation-plots')
        if backup.exists():
            raise ValueError(f'Backup already exists: {backup}; no files changed')
        shutil.copy2(source, backup)
        shutil.copy2(supplied, header)
        source.write_text(new)
    elif not header.exists():
        shutil.copy2(supplied, header)
    print('Radiation slice export installed. Rebuild octotiger before running.')

if __name__ == '__main__':
    try: main()
    except (OSError, ValueError) as e: sys.exit(str(e))
