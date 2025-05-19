import setuptools 

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="efficient-classifier",
    version="1.0.0",
    author="Javier D. Segura",
    author_email="javier.dominguez.segura@gmail.com",
    description="A library for shallow and deep classifiers. Shows pipeline DAG, communicates with SlackBot stores results in database",
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