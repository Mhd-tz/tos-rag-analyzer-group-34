# ToS RAG Analyzer (Group 34)

**IAT 360 - Designing the Digital Future**  
*Simon Fraser University*

## Academic Context
This project was created for IAT 360 at Simon Fraser University.
It demonstrates a production-ready RAG system for privacy policy analysis.

**Deployed App**: https://tos-rag-analyzer-group-34-jkokhkvvcrtdcpghfhs7re.streamlit.app/

*Note*: The app is deployed on Streamlit Cloud and may take a few seconds to load. If it does not load or the app is paused, please run it locally using `streamlit run app.py`.

## Project Overview
This project is an implementation of **Option 2: Deploy Existing Model (RAG)**. It designed to make privacy policies and Terms of Service (ToS) transparent and easy to understand. It uses **Retrieval Augmented Generation (RAG)** to ground its answers in real legal documents, reducing hallucinations and providing accurate citations.

### How It Works
1.  **Knowledge Base**: We aggregated a dataset of privacy policies (OPP-115 + others) and chunked them into small text segments.
2.  **Vector Database**: These chunks are converted into mathematical vectors (embeddings) using a Hugging Face model (`all-MiniLM-L6-v2`) and stored in a FAISS index.
3.  **Retrieval**: When a user asks a question, the system finds the most relevant chunks from the database.
4.  **Generation**: The system sends the question + relevant chunks to an LLM (GPT-3.5 or Hugging Face), which generates a plain-English answer based *only* on those documents.

---

## Getting Started

### Prerequisites
- Python 3.9+
- An OpenAI API Key (optional)
- A Hugging Face Token

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Mhd-tz/tos-rag-analyzer-group-34.git
    cd tos-rag-analyzer-group-34
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---

## How to Test the App

### 1. Knowledge Base (General Trends)
Use the **"Knowledge Base"** tab to ask general questions about industry practices.
*   *Example*: "What is the standard data retention period?"
*   *Goal*: See how the model synthesizes information from 100+ policies.

### 2. Live Analysis (Specific Audit)
Use the **"Live Analysis"** tab to audit a specific company (e.g., Google, TikTok, or a startup).
1.  Paste the URL or text of a privacy policy.
2.  Click **"Process Policy"**.
3.  Ask specific questions like: "Does this policy allow AI training on my data?"
*   *Goal*: Verify the model can handle *new, unseen* data instantly.

### 3. Dual LLM Comparison
Toggle between **OpenAI** and **Hugging Face** in the sidebar to compare performance.
*   *OpenAI*: Faster, more nuanced legal reasoning.
*   *Hugging Face*: Free, open-source, runs locally (slower).

---

## Project Structure
- `app.py`: Main application code (Streamlit).
- `requirements.txt`: Python dependencies.
- `faiss_index_tos_hf/`: The vector database folder.
- `notebooks/`: Contains the Jupyter notebook for data processing.
