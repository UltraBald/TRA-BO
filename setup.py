from setuptools import find_packages, setup


setup(
    name="tra-bo",
    version="0.1.0",
    description="Temporal Rank Agreement for time-series hyperparameter optimization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"tra_bo.prompts": ["*.txt"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
    ],
    entry_points={"console_scripts": ["tra-bo=tra_bo.cli:main"]},
)
