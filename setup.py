from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jitmind",
    version="0.1.0",
    author="Divyam Talwar",
    description=(
        "Just-in-time memory system for AI agents with self-editing bi-temporal memory, "
        "temporal graph reasoning, and an iterative Plan->Search->Integrate->Reflect loop."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DivyamTalwar/JITMIND",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jitmind-eval=eval.run:main",
        ],
    },
)
