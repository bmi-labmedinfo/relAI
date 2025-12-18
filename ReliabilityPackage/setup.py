from setuptools import find_packages, setup

with open("Readme.md", "r") as f:
    long_description = f.read()

setup(
    name="ReliabilityPackage",
    version="0.0.20",
    description="A tool to compute the reliability of Machine Learning predictions",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bmi-labmedinfo/Reliability",
    author="LorenzoPeracchio",
    author_email="lorenzo.peracchio01@universitadipavia.it",
    license="Creative Commons Attribution-NonCommercial 4.0 International",
    classifiers=[
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    install_requires=["numpy", "scikit-learn", "torch", "plotly", "matplotlib"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"]
    },
    python_requires=">=3.6"
)
