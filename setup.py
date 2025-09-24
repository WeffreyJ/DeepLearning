from setuptools import setup, find_packages

setup(
    name="dlrepo",
    version="0.1.0",
    description="Jeffrey's deep learning playbook (config-driven PyTorch)",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[],
    python_requires=">=3.9",
)
