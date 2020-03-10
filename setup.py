from setuptools import find_packages, setup

requirements = []
with open("requirements.txt", "r") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        requirements.append(line)

setup(
    name='nnga',
    version='0.1.0',
    packages=find_packages(),
    entry_points=dict(console_scripts=["nnga = nnga.nnga_cli:main"]),
    description="Neural Network Genetic Algorithm",
    author="Rafael Silva Del Lama",
    author_email="rafael.lama@usp.br",
    install_requires=requirements,
    tests_require=["pytest", "pytest-flake8", "pytest-cov"],
)
