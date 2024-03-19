from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        requirements = [line.strip() for line in lines]
    return requirements


setup(
    name="medseg",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    author="snubhcvc",
    author_email="whikwon@gmail.com",
    description="Medical image segmentation",
    url="https://github.com/SNUBH-CVC/medseg",
)
