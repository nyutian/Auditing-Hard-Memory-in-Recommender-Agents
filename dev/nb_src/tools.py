"""Dev helpers for the notebook source kept in percent format.

  python tools.py run   <project_dir> [part ...]   run the parts as one script, printing cell results
  python tools.py build <project_dir> <out.ipynb> [part ...]   assemble the notebook
  python tools.py exec  <project_dir> <nb.ipynb>   execute the notebook in place
"""
import ast, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PARTS = ["p1_core.py", "p2_writers.py", "p3_experiments.py", "p4_update_fig.py"]


def cells_of(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    out = []
    for kind, buf in cells:
        while buf and not buf[0].strip():
            buf.pop(0)
        while buf and not buf[-1].strip():
            buf.pop()
        if kind == "markdown":
            buf = [l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in buf]
        if buf:
            out.append((kind, "\n".join(buf)))
    return out


def load(parts):
    cells = []
    for p in parts:
        f = HERE / p
        if f.exists():
            cells += cells_of(f.read_text())
    return cells


def run(project, parts):
    os.chdir(project)
    os.environ.setdefault("MPLBACKEND", "Agg")
    ns = sys.modules["__main__"].__dict__      # like a notebook kernel: cells live in __main__
    for n, (kind, src) in enumerate(load(parts)):
        if kind != "code":
            continue
        t0 = time.time()
        tree = ast.parse(src)
        last = tree.body[-1] if tree.body else None
        if isinstance(last, ast.Expr):
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body, f"<cell {n}>", "exec"), ns)
            val = eval(compile(ast.Expression(last.value), f"<cell {n}>", "eval"), ns)
            if val is not None:
                try:
                    import pandas as pd
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        print(val.to_string())
                    else:
                        print(repr(val))
                except Exception:
                    print(repr(val))
        else:
            exec(compile(tree, f"<cell {n}>", "exec"), ns)
        dt = time.time() - t0
        if dt > 1.0:
            print(f"   [cell {n}: {dt:.1f}s]")
    return ns


def build(project, out, parts):
    import nbformat
    nb = nbformat.v4.new_notebook()
    for kind, src in load(parts):
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, str(Path(project) / out))
    print("wrote", Path(project) / out, "| cells:", len(nb.cells))


def execute(project, name):
    import nbformat
    from nbclient import NotebookClient
    path = Path(project) / name
    nb = nbformat.read(str(path), as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(project)}}).execute()
    nbformat.write(nb, str(path))
    print(f"executed {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd, project = sys.argv[1], sys.argv[2]
    if cmd == "run":
        run(project, sys.argv[3:] or DEFAULT_PARTS)
    elif cmd == "build":
        build(project, sys.argv[3], sys.argv[4:] or DEFAULT_PARTS)
    elif cmd == "exec":
        execute(project, sys.argv[3])
