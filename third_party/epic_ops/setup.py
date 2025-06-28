import os
import sys
import subprocess
import sysconfig
from pathlib import Path
from setuptools import find_packages, setup

import torch

from tools.setup_helpers.cmake import CMake
from tools.setup_helpers.env import IS_WINDOWS

PACKAGE_NAME = "epic-ops"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 11], "Requires PyTorch >= 1.11"


def get_version() -> str:
    cwd = Path(__file__).parent

    with open(cwd / "version.txt") as f:
        version = f.readline().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        sha = "Unknown"

    if os.getenv("STOPS_BUILD_VERSION"):
        version = os.getenv("STOPS_BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    # write version.py
    with open(cwd / "version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")

    return version


def build_extension(version: str) -> None:
    cmake = CMake()

    # CMAKE: full path to python library
    if IS_WINDOWS:
        cmake_python_library = "{}/libs/python{}.lib".format(
            sysconfig.get_config_var("prefix"),
            sysconfig.get_config_var("VERSION"))
        # Fix virtualenv builds
        # TODO: Fix for python < 3.3
        if not os.path.exists(cmake_python_library):
            cmake_python_library = "{}/libs/python{}.lib".format(
                sys.base_prefix,
                sysconfig.get_config_var("VERSION"))
    else:
        cmake_python_library = "{}/{}".format(
            sysconfig.get_config_var("LIBDIR"),
            sysconfig.get_config_var("INSTSONAME"))
    print("cmake_python_library: ", cmake_python_library)
    cmake.generate(
        version=version,
        cmake_python_library=cmake_python_library,
        cmake_torch_library=torch.utils.cmake_prefix_path,
        rerun=False,
    )
    cmake.build()


if __name__ == "__main__":
    version = get_version()

    print(f"Building {PACKAGE_NAME}-{version}")

    build_extension(version)

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="Haoran Geng & Ming Yang",
        author_email="ghr@stu.pku.edu.cn",
        url="https://github.com/geng-haoran/epic_ops",
        download_url="https://github.com/geng-haoran/epic_ops/tags",
        description="Epic Ops",
        long_description=Path("README.md").read_text(),
        packages=find_packages(exclude=("tests",)),
        package_data={"epic_ops": ["*.dll", "*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.8",
    )
