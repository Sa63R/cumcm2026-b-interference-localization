"""Reproduce the visibility counterexample against the original V4 ZIP.

Usage:
    python audit_v4.py /path/to/B题第四问_V4优化版.zip

Only extracts the Python files into a temporary directory; never edits V4.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
from radius_direction import visibility_probability

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    args=p.parse_args()
    if not args.archive.is_file():p.error('Archive not found')
    digest=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.archive) as z:
            for name in z.namelist():
                # This package has flat Python module names. Do not extract
                # arbitrary archive paths or execute non-Python archive content.
                if '/' not in name and '\\' not in name and name.endswith('.py'):
                    Path(tmp,name).write_bytes(z.read(name))
        sys.path.insert(0,tmp)
        module=importlib.import_module('q4_information')
        old=module.heuristic_visibility((0.,0.),[(500.,0.)],[(1200.,0.)],(1300.,0.))
        new=visibility_probability((0.,0.),[(500.,0.)],[(1200.,0.)],(1300.,0.))
        print(json.dumps({'archive_sha256':digest,'old_heuristic':old,'new_marginal':new,
                          'scope':'fixed candidate position; inference demonstration, not an episode benchmark'},indent=2))

if __name__=='__main__':main()
