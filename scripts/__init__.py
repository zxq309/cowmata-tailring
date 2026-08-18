"""Thin command-line shims.

Every module here is a four-line wrapper around a library function in
:mod:`cowmata`.  The 20260818 versions each carried their own argument parser
and their own copy of the label list; the logic now lives in the package so it
can be imported, tested and called without a subprocess.
"""
