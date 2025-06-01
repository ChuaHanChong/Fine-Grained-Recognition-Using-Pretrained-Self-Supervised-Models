from setuptools import setup, find_packages

PACKAGE_NAME = "src"

# Define the package
setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    python_requires=">=3.10",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={"": "src"},
)