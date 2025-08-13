from setuptools import setup, find_packages

setup(
    name='akm',  
    version='1.0',  
    packages=find_packages(),  
    install_requires=[
        'opencv-python-headless',
        'sapien==2.2.2',
        'transforms3d',
        'trimesh',
        "mplib==0.2.1",
        "pyglet<2"
    ],
)
