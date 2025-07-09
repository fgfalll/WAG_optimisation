from .grdecl_writer import write_grdecl

__all__ = [
    "write_grdecl"
]

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())