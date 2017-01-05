import re
import setuptools
import sys


if not sys.version_info >= (3, 5):
    exit("Sorry, Python must be later than 3.5.")


setuptools.setup(
    name="tensorflow-extenteten",
    version=re.search(r'__version__ *= *"([0-9]+\.[0-9]+\.[0-9]+)" *\n',
                      open("extenteten/__init__.py").read()).group(1),
    description="TensorFlow extention library",
    long_description=open("README.md").read(),
    license="Public Domain",
    author="Yota Toyama",
    author_email="raviqqe@gmail.com",
    url="https://github.com/raviqqe/tensorflow-extenteten/",
    packages=["extenteten"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Public Domain",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
    ],
)
