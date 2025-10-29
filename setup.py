from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ayyad_apis",
    version="0.1.4",
    author_email="ahmedyad200@gmail.com",
    description="Collection of Python wrappers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ahmed Ayyad",
    url="https://github.com/ahmedayyad-dev/Ayyad-APIs",
    packages=find_packages(),
    install_requires=[
        "aiohttp",
        "aiofiles"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
