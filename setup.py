import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requires = f.read().splitlines()

setuptools.setup(
    name="pefty_llama",
    version="0.0.1