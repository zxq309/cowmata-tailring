"""Experiments that are not part of the deployed pipeline.

Code lands here when it has been tried, recorded and *not* adopted.  Keeping it
in the core package implied it was on the delivery path; deleting it would have
thrown away the evidence of what was tested.

- ``fusion.py``: logit-space late fusion of the deep and hand-crafted branches.
  The 20260818 report concluded that fusion did not beat the better single
  branch, so it is not in ``cowmata/`` any more. It stays here because the next
  round of data may change that conclusion, and re-deriving the weight search
  would be wasted work.
"""
