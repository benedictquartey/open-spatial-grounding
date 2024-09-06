from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="open_spatial_grounding",
    version="0.1.0",
    author="Benedict Quartey",
    author_email="benedict_quartey@brown.edu",
    description="Out of the box package for grounding open vocabulary objects with spatial constraints",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/benedictquartey/open-spatial-grounding",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        # Add your project's dependencies here
        "numpy"
        "open3d"
        "networkx "
        "pillow"
        "matplotlib"
        "transformers"
        "torch"
        "torchvision"
        "tqdm"
        "ipykernel"
        "timm==0.9.10"
        "fire"
        "more_itertools"
        "opencv-python==4.9.0.80"
        "pyliblzfse"
    ],
)
