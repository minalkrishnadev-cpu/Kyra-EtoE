"""
Run once from src/fit:   python patch_tf_compat.py

HMR was written for TensorFlow 1.3, which accepted
    tf.name_scope("name", [x])      tf.variable_scope("name", [x])
TensorFlow 1.15 treats the 2nd positional argument as default_name (a string) and
raises TypeError. This rewrites those calls to use values=[...] and keeps a
backup copy (<file>.orig) of every file it changes. Safe to run more than once.
"""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
PATTERN = re.compile(r"\b(name_scope|variable_scope)\(\s*([^,()\[\]]+?)\s*,\s*\[")


def patch_text(text):
    return PATTERN.subn(lambda m: "%s(%s, values=[" % (m.group(1), m.group(2)), text)


def main():
    total = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            new, n = patch_text(text)
            if n:
                if not os.path.exists(path + ".orig"):
                    with open(path + ".orig", "w", encoding="utf-8") as f:
                        f.write(text)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
                print("patched %d call(s) in %s" % (n, os.path.relpath(path, ROOT)))
                total += n
    print("done: %d call(s) patched" % total)


if __name__ == "__main__":
    main()
