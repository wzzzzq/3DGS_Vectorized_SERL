from setuptools import setup, find_packages

setup(
    name="mobile_robot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "mujoco",
        "numpy",
        "scipy",
    ],
    description="Mobile robot environment for SERL",
)
