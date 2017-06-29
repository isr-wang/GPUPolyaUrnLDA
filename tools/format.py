#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""format.py: Tool to format or tidy files using clang-format and clang-tidy"""

import argparse
import os
import re
import subprocess


__author__ = "Kunal Sarkhel"
__copyright__ = "Copyright 2017, Kunal Sarkhel"
__credits__ = ["Kunal Sarkhel"]
__license__ = "Apache-2.0"
__version__ = "0.0.0"
__maintainer__ = "Kunal Sarkhel"
__email__ = "ksarkhel@gmu.edu"
__status__ = "Alpha"


def find_cpp_files(paths, exclude_pattern=None):
    """Recursively walk a list of directories and return the list of C++ files.

    :paths: list of strings representing directories
    :exclude_pattern: regular expression string for excluding paths
    :returns: set of strings representing absolute paths to C++ files

    """
    abspath = lambda root, path: os.path.abspath(os.path.join(root, path))
    if exclude_pattern:
        exclude_pattern = re.compile(exclude_pattern)

    matches = set()
    cpp_extensions = ('.cu', '.cxx', '.h', '.hxx')
    for path in paths:
        if not os.path.isdir(path):
            raise SystemExit('{} is not a directory!'.format(path))
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith(cpp_extensions):
                    pathname = abspath(root, filename)
                    if exclude_pattern and not exclude_pattern.match(pathname):
                        matches.add(pathname)
                    elif not exclude_pattern:
                        matches.add(pathname)
    return matches


def format_cpp_files(filenames):
    """Format a list of C++ files using clang-format

    :filenames: list of strings representing filenames to be formatted
    :returns: None

    """
    for filename in filenames:
        print('Formatting {}'.format(filename))
        subprocess.Popen('clang-format -i {} -style=file'.format(filename),
                         shell=True).wait()


def tidy_cpp_files(filenames):
    """Tidy a list of C++ files using clang-tidy

    :filenames: list of strings representing filenames to be tidied
    :returns: None

    """
    raise SystemExit('clang-tidy integration is not yet implemented')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to format or tidy files using clang-format and clang-tidy')
    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument('--format', '-f', action='store_const', dest='format',
                    default=False, const=True,
                    help='Format the files using clang-format')
    gp.add_argument('--tidy', '-t', action='store_const', dest='tidy',
                    default=False, const=True,
                    help='tidy the files using clang-tidy')
    parser.add_argument('--exclude-pattern', action='store',
                        dest='exclude_pattern',
                        help='Regular expression for excluding paths')
    parser.add_argument('DIRECTORY', nargs='*', default=[os.getcwd()],
                        help='Directory to search for C++ files')
    args = parser.parse_args()

    cpp_files = find_cpp_files(args.DIRECTORY,
                            exclude_pattern=args.exclude_pattern)
    if args.format:
        format_cpp_files(cpp_files)
    elif args.tidy:
        tidy_cpp_files(cpp_files)
