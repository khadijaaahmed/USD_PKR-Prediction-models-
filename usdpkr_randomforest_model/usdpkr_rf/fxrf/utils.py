"""Small helper functions used by several modules."""
import os


def unique_path(folder, name, ext):
    """Never overwrite: if the file exists, add _2, _3 ... to the name."""
    p = os.path.join(folder, f"{name}{ext}")
    k = 2
    while os.path.exists(p):
        p = os.path.join(folder, f"{name}_{k}{ext}")
        k += 1
    return p


def print_table(headers, rows, indent="  "):
    """Column form: headers on top, values underneath."""
    rows = [[str(x) for x in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]
    print(indent + "".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print(indent + "".join("-" * (w[i] - 1) + " " for i in range(len(headers))))
    for r in rows:
        print(indent + "".join(r[i].ljust(w[i]) for i in range(len(headers))))
