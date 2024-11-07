from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bestrag",
    version="0.2.0",
    description="BestRAG (Best Retrieval Augmented) is a library for storing and"
    " searching document embeddings in a Qdrant vector database. It uses a "
    "hybrid embedding technique combining dense, late interaction and sparse representations for better performance.",
    author="samadpls",
    author_email="abdulsamadsid1@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samadpls/bestRAG",
    packages=find_packages(),
    install_requires=[
        "fastembed==0.4.1",
        "streamlit",
        "pytest",
        "flake8",
        "PyPDF2",
        "qdrant-client",
        "onnxruntime==1.19.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
