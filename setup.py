from setuptools import setup, find_packages

setup(
    name='embodied_analogy',  
    version='1.0',  
    packages=find_packages(),  
    install_requires=[
        'open3d',
        'opencv-python',
        'pygame==2.6.1',
        'sapien==3.0.0b1',
        'transforms3d',
        'trimesh',
        "ftfy",
        "regex"
    ],
    author='Boyuan Zhang',
    author_email='zhangboyuan17@mails.ucas.ac.cn',
)
