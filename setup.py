from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="hindi_tokenizer",
    version="0.1.0",
    author="Abhishek Dhiman",
    author_email="your.email@example.com",
    description="A SentencePiece tokenizer optimized for Hindi language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhishekmaroon5/hindi_tokenizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Hindi",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hindi_tokenizer=hindi_tokenizer.cli:main",
        ],
    },
)
