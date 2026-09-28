"""Execute this experiment's Python cells without a TCP Jupyter kernel.

Usage: python3 notebooks/execute_in_process.py notebooks/spx_no_mixup_comparison.ipynb
Uses IPython and the inline Matplotlib backend to retain rich outputs. This is
not a substitute for testing arbitrary notebooks that need a kernel or widgets.
"""
import argparse
import os
from pathlib import Path

import matplotlib
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    path = parser.parse_args().notebook.resolve()
    nb = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    backend = 'module://matplotlib_inline.backend_inline'
    matplotlib.use(backend)
    configure_inline_support(shell, backend)
    os.chdir(path.parent)
    count = 0
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'Executing code cell {count} (notebook cell {index + 1})', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source)
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout)
            print(captured.stderr)
            raise RuntimeError(f'Execution failed at notebook cell {index + 1}') from (
                result.error_before_exec or result.error_in_exec)
        outputs = []
        for name, content in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if content:
                outputs.append(nbformat.v4.new_output('stream', name=name, text=content))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        cell.outputs = outputs
        cell.execution_count = count
    nb.metadata['execution_method'] = 'IPython in-process; no TCP kernel; cells executed sequentially'
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(f'Saved {count} executed cells to {path}')


if __name__ == '__main__':
    main()
