# ⚖️ ToS RAG Analyzer (Group 34)

**IAT 360 - Designing the Digital Future**  
*Simon Fraser University*

## 📖 Project Overview
This project is my implementation of **Option 2: Deploy Existing Model (RAG)**. I designed it to make privacy policies and Terms of Service (ToS) transparent and easy to understand. It uses **Retrieval Augmented Generation (RAG)** to ground its answers in real legal documents, reducing hallucinations and providing accurate citations.

### How It Works ⚙️
1.  **Knowledge Base**: I aggregated a dataset of privacy policies (OPP-115 + others) and chunked them into small text segments.
2.  **Vector Database**: These chunks are converted into mathematical vectors (embeddings) using a Hugging Face model (`all-MiniLM-L6-v2`) and stored in a FAISS index.
3.  **Retrieval**: When a user asks a question, the system finds the most relevant chunks from the database.
4.  **Generation**: The system sends the question + relevant chunks to an LLM (GPT-3.5 or Hugging Face), which generates a plain-English answer based *only* on those documents.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An OpenAI API Key (for the recommended model)
- A Hugging Face Token (optional, for the free model)

### Installation
1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
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

## 🧪 How to Test the App

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

## 🛠️ For Developers: Adding More Datasets

If you want to expand the knowledge base (e.g., add GDPR documents or new policies), follow these steps:

1.  **Open the Notebook**:
    Open `notebooks/IAT360_TOS_RAG_Final.ipynb` in Jupyter or Google Colab.

2.  **Add Your Dataset**:
    Scroll to **Step 1: LOADING DATASETS**. You will see code like this:
    ```python
    # Dataset 1: OPP-115
    dataset_1 = load_dataset("alzoubi36/opp_115")
    
    # Dataset 2: PrivacyPolicy
    dataset_2 = load_dataset("sjsq/PrivacyPolicy")
    ```
    Simply add your new dataset:
    ```python
    # Dataset 3: Your New Dataset
    dataset_3 = load_dataset("your-huggingface-dataset-id")
    ```

3.  **Process the New Data**:
    Scroll to **Step 2: PREPARING DOCUMENTS**. Copy the processing loop for `dataset_2` and adapt it for `dataset_3`. Ensure you map the text column correctly (e.g., `item['text']` or `item['content']`).

4.  **Run the Notebook**:
    Run all cells. This will:
    - Download the new data.
    - Chunk it.
    - Create a new FAISS index.
    - Save a new `faiss_index_tos_hf.zip` file.

5.  **Update the App**:
    - Download the new `faiss_index_tos_hf.zip`.
    - Unzip it and replace the `faiss_index_tos_hf` folder in this project directory.
    - Restart the Streamlit app.

---

## 📂 Project Structure
- `app.py`: Main application code (Streamlit).
- `requirements.txt`: Python dependencies.
- `faiss_index_tos_hf/`: The vector database folder.
- `notebooks/`: Contains the Jupyter notebook for data processing.
