
# RAG_TEST: Retrieval-Augmented Generation Project

## Features

- Standard RAG pipeline: PDF/TXT document loading, chunking, OpenAI embedding, FAISS vector search, and answer generation.
- Self-RAG emulation: Iterative answer generation with LLM-based critique and self-reflection.
- Special token critique: LLM outputs RETRIEVE, ISSUP, ISREL, ISUSE sections for each answer.
- Adaptive stopping: The loop stops early if ISSUP, ISREL, and ISUSE scores meet configurable thresholds.
- Easy configuration: Thresholds and max iterations are set at the top of `Self_rag.py` for quick tuning.





## 1. Installing Python Dependencies

It is recommended to use a Python virtual environment to avoid dependency conflicts.

### On Windows

1. Make sure you have Python 3.12+ installed:
   ```sh
   python --version
   ```
   If not, download and install from [python.org](https://www.python.org/downloads/).
2. Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. To deactivate the venv when done:
   ```sh
   deactivate
   ```

### On Mac (macOS) or Linux

1. Make sure you have Python 3.12+ installed:
   ```sh
   python3 --version
   ```
   If not, install it via Homebrew (Mac):
   ```sh
   brew install python@3.12
   ```
2. Create and activate a virtual environment:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. To deactivate the venv when done:
   ```sh
   deactivate
   ```

> **Note:** Python 3.12+ is fully supported for the custom Self-RAG pipeline in this repo. The official self-rag library may require Python 3.10/3.11 due to torch version compatibility (see below).

---

## 2. Setting the OpenAI API Key

You must set your OpenAI API key as an environment variable before running the script.

### On Windows

```sh
set OPENAI_API_KEY=sk-...
```

### On Mac (macOS) or Linux

```sh
export OPENAI_API_KEY=sk-...
```

---


1. Place your PDF or TXT files in `RAG_TEST/Documents/`.
2. (Recommended) Create and activate a virtual environment (see above).
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key:
   ```sh
   set OPENAI_API_KEY=sk-...   # Windows
   export OPENAI_API_KEY=sk-... # Linux/Mac
   ```
5. Run the Self-RAG script:
   ```sh
   python RAG_TEST/SELF_RAG/Self_rag.py
   ```
6. Ask questions interactively. Type `exit` to quit.

---

## Configuration

Edit the following parameters at the top of `RAG_TEST/SELF_RAG/Self_rag.py`:

```python
# Self-RAG adaptive stopping config (edit here)
ISSUP_THRESHOLD = 0.8  # Minimum support score (0-1) to stop
ISREL_THRESHOLD = 0.8  # Minimum relevance score (0-1) to stop
ISUSE_THRESHOLD = 0.8  # Minimum usefulness score (0-1) to stop
MAX_ITERATIONS = 3     # Maximum RAG-reflection cycles
```

---


This project includes a standard RAG pipeline and a Self-RAG implementation (custom, LLM-based) that mimics the [self-rag](https://github.com/AkariAsai/self-rag) library's behavior. The default script does not require the official self-rag library, but you can optionally install and experiment with it (see below).

---


## Self-RAG Adaptive Stopping (How it works)

After each answer, the LLM critiques itself using four special tokens:

- **RETRIEVE:** What information was retrieved and used?
- **ISSUP:** On a scale from 0 to 1, how well is the answer supported by the retrieved context?
- **ISREL:** On a scale from 0 to 1, how relevant is the answer to the question and context?
- **ISUSE:** On a scale from 0 to 1, how useful and actionable is the answer for the user?

If all three numeric scores (ISSUP, ISREL, ISUSE) meet or exceed their thresholds, the loop stops early. Otherwise, it continues up to `MAX_ITERATIONS`.

---

> **Note:** The `self-rag` library requires `torch==2.1.2`, which is not available for Python 3.12. You may need to use Python 3.10/3.11 for full compatibility. If you proceed with Python 3.12, you may need to adjust the requirements or install a compatible torch version manually.

### Steps:

1. **Clone the Self-RAG repository:**
   ```sh
   git clone https://github.com/AkariAsai/self-rag.git
   ```
2. **Navigate to the repo:**
   ```sh
   cd self-rag
   ```
3. **Install requirements:**
   ```sh
   pip install -r requirements.txt
   ```
   - If you see an error about `torch==2.1.2`, try installing a compatible version of torch for your Python version, or use Python 3.10/3.11.

---

## Installation
To install all dependencies, run:

```sh
pip install -r requirements.txt
```

## Troubleshooting

- If you see errors like `No matching distribution found for torch==2.1.2`, downgrade your Python version to 3.10 or 3.11 and try again.
- For Windows, you may need to use the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to get a compatible wheel.

---


## Project Structure

- `RAG_TEST/` - Main project directory
- `RAG_TEST/SELF_RAG/Self_rag.py` - Main Self-RAG implementation (custom, LLM-based)
- `self-rag/` - Cloned official Self-RAG repository

---


## Next Steps

- (Optional) Update your `Self_rag.py` to use the official `self-rag` library for a true Self-RAG pipeline.
- Print special tokens during execution as needed (already implemented in the custom script).

---

## Example Error Output

```
ERROR: Could not find a version that satisfies the requirement torch==2.1.2 (from versions: ...)
ERROR: No matching distribution found for torch==2.1.2
```

---

## Useful Links
- [Self-RAG GitHub](https://github.com/AkariAsai/self-rag)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

---

## Installing Self-RAG
To install the official Self-RAG library and its dependencies:

1. Clone the repository:
   ```sh
   git clone https://github.com/AkariAsai/self-rag.git
   ```
2. Change directory:
   ```sh
   cd self-rag
   ```
3. Install requirements (see torch note below):
   ```sh
   pip install -r requirements.txt
   ```

If you encounter errors with `torch==2.1.2` on Python 3.12+, edit `self-rag/requirements.txt` and change the torch line to a compatible version (e.g., `torch==2.2.0`).

# If you encounter errors with torch==2.1.2 on Python 3.12+, try this workaround:
# 1. Edit self-rag/requirements.txt and change the torch line to:
#    torch==2.2.0
# 2. Then run:
#    pip install -r self-rag/requirements.txt
#
# Note: This may cause compatibility issues with self-rag. If you see errors, consider using Python 3.10/3.11 for best results.


---

## FAQ

**Q: How do I change the stopping criteria?**
A: Edit the threshold variables at the top of `Self_rag.py`.

**Q: Can I use this with my own documents?**
A: Yes! Place your PDFs or TXTs in `RAG_TEST/Documents/` and rerun the script.

**Q: Do I need the official self-rag library?**
A: No, the main script works out of the box. The official library is optional for advanced users.

**Q: How do I get more verbose output?**
A: The script prints all answers, critiques, and token usage by default.

---

For further help, please open an issue or contact the maintainer.
