"""The compiler's targets, a folder each holding the target's runtime model
(``trainium`` also its NKI translation).  The lowering in ``codegen`` is
shared by every target; ``Backend`` (``backend.py``) holds a target's hooks
into it.
"""
