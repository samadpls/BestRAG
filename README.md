
<img src="https://github.com/user-attachments/assets/e23d11d5-2d7b-44e2-aa11-59ddcb66bebc" align=left height=150px>

![Supported python versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-black.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](LICENSE)
[![Run Pytest](https://github.com/samadpls/BestRAG/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/samadpls/BestRAG/actions/workflows/pytest.yml)
![GitHub stars](https://img.shields.io/github/stars/samadpls/BestRAG?color=red&label=stars&logoColor=black&style=social)
![PyPI - Downloads](https://img.shields.io/pypi/dm/bestrag?style=social)



Introducing **BestRAG**! This Python library leverages a hybrid Retrieval-Augmented Generation (RAG) approach to efficiently store and retrieve embeddings. By combining dense, sparse, and late interaction embeddings, **BestRAG** offers a robust solution for managing large datasets.


## ✨ Features

🚀 **Hybrid RAG**: Utilizes dense, sparse, and late interaction embeddings for enhanced performance.  
🔌 **Easy Integration**: Simple API for storing and searching embeddings.  
📄 **PDF Support**: Directly store embeddings from PDF documents.  

## 🚀 Installation

To install **BestRAG**, simply run:

```bash
pip install bestrag
```

## 📦 Usage

Here’s how you can use **BestRAG** in your projects:

```python
from bestrag import BestRAG

rag = BestRAG(
    url="https://YOUR_QDRANT_URL", 
    api_key="YOUR_API_KEY", 
    collection_name="YOUR_COLLECTION_NAME"
)

# Store embeddings from a PDF
rag.store_pdf_embeddings("your_pdf_file.pdf")

# Search using a query
results = rag.search(query="your search query", limit=10)
print(results)
```

> **Note**: Qdrant offers a free tier with 1GB of storage. To generate your API key and endpoint, visit [Qdrant](https://qdrant.tech/).

## 🤝 Contributing

Feel free to contribute to **BestRAG**! Whether it’s reporting bugs, suggesting features, or submitting pull requests, your contributions are welcome. 

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

Created by [samadpls](https://github.com/samadpls) 🎉
