import setuptools 
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Default requirements in case requirements.txt is not found
default_requirements = [
    "numpy>=2.0.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.0.0",
    "seaborn>=0.11.0",
    "tensorflow>=2.0.0",
    "torch>=1.0.0",
    "scipy>=1.0.0",
    "slack_bolt>=1.0.0",
    "slack_sdk>=3.0.0",
    "joblib>=1.0.0"
]

# Try to read requirements from requirements.txt
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = default_requirements

setuptools.setup(
    name="efficient-classifier",
    version="2.1.2",
    author="Javier D. Segura",
    author_email="jdominguez.ieu2023@student.ie.edu",
    description="Dataset-agnostic ML classification library. Visualization tools, Slack integration, support for multiple-pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/javidsegura/efficient-classifier",
    packages=setuptools.find_packages(),
    package_data={
        "efficient_classifier": [
            "test/*",
            "configurations.yaml"  
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
)