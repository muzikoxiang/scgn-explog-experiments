"""Compact BasicSR package bundled by SPAN.

The upstream snapshot contains only the components needed by SPAN. Avoid
eagerly importing every optional BasicSR subsystem when the package is loaded;
the train/test entry points import and register their required modules.
"""

try:
    from .version import __gitsha__, __version__
except ImportError:  # setup.py generates version.py during installation
    __version__ = '1.3.5'
    __gitsha__ = 'unknown'

__all__ = ['__gitsha__', '__version__']
