from setuptools import setup, find_packages

setup(
    name="mle-dev",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "six",
    ],
    entry_points={
        "console_scripts": [
            "ingest-data=src.ingest_data:fetch_housing_data",
            "train-model=src.train:train_model",
            "score-model=src.score:score_model",
        ]
    },
)
