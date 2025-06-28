import os

import torch

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass


def _get_extension_path(lib_name):
    lib_dir = os.path.dirname(__file__)
    if os.name == "nt":
        os.add_dll_directory(lib_dir)

    import importlib.machinery
    loader_details = (importlib.machinery.ExtensionFileLoader,
                      importlib.machinery.EXTENSION_SUFFIXES)
    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError
    return ext_specs.origin


try:
    torch.ops.load_library(_get_extension_path("libepic_ops"))
except (ImportError, OSError) as e:
    print(e)
    pass
