from setuptools import setup, find_packages
setup(
    name="dlrepo",
    version="0.1.0",
    description="Deep learning playbook (config-driven PyTorch)",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires=">=3.9",
)
