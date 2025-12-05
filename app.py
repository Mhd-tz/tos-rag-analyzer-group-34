import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Optional, Mapping
from huggingface_hub import InferenceClient
import requests
from bs4 import BeautifulSoup # Tutorial used for scraping: https://realpython.com/beautiful-soup-web-scraper-python/
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pydantic.v1.main

# ============================================
# Got this fix from: https://github.com/pydantic/pydantic/discussions/6766
# ============================================
# This fixes "KeyError: '__fields_set__'" when loading FAISS index
# created with Pydantic v2 in an environment using Pydantic v1.
original_setstate = pydantic.v1.main.BaseModel.__setstate__

def patched_setstate(self, state):
    if '__fields_set__' not in state:
        state['__fields_set__'] = set(state.keys())
    return original_setstate(self, state)

pydantic.v1.main.BaseModel.__setstate__ = patched_setstate

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="ToS Analyzer - Hybrid Edition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF512F;
        margin-bottom: 1rem;
        display: inline-block;
    }
    .emoji-header { font-size: 3rem; vertical-align: bottom; }
    .disclaimer {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        color: #000;
    }
    h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div style='text-align: center;'>
    <span class='main-header'>ToS Analyzer</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem;'>Understand Terms of Service in Plain English</p>
    <p style='color: #666;'>Hybrid Architecture: Hugging Face (Embeddings) + OpenAI (Reasoning)</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer as we discussed in the our presentation to have a clear understanding of the tool
st.markdown("""
<div class="disclaimer">
    <strong>Important Disclaimer</strong><br>
    This tool provides AI-generated interpretations. It is <strong>NOT legal advice</strong>. 
    Always read the full Terms of Service and consult a qualified attorney for legal decisions.
    <br>
    (Disclaimer as we discussed in our presentation to have a clear understanding of the tool)
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("Authentication")
    
    # Model Selection
    model_provider = st.radio(
        "Select Model Provider:",
        ["OpenAI (Recommended)", "Hugging Face (Free)"],
        help="OpenAI is faster/smarter. Hugging Face is free but slower."
    )
    
    api_key = ""
    
    if model_provider == "OpenAI (Recommended)":
        st.markdown("""
        **Get your OpenAI API Key:**
        1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        2. Create new secret key
        3. Paste below
        """)
        with st.form("openai_key_form"):
            api_key_input = st.text_input("OpenAI API Key", type="password")
            submitted = st.form_submit_button("Connect")
            if submitted:
                api_key = api_key_input
            elif os.environ.get("OPENAI_API_KEY"):
                api_key = os.environ["OPENAI_API_KEY"]
                
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("Connected to OpenAI!")
        else:
            st.warning("Waiting for API Key...")
            
    else: # Hugging Face
        st.markdown("""
        **Get your Hugging Face Token:**
        1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        2. Create new token (Read access)
        3. Paste below
        """)
        with st.form("hf_token_form"):
            api_key_input = st.text_input("Hugging Face Token", type="password")
            submitted = st.form_submit_button("Connect")
            if submitted:
                api_key = api_key_input
            elif os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
                api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            st.success("Connected to Hugging Face!")
        else:
            st.warning("Waiting for Token...")
            
        hf_model_id = st.selectbox(
            "Select Free Model:",
            [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "HuggingFaceH4/zephyr-7b-beta"
            ]
        )
    
    st.markdown("---")
    
    st.header("Settings")
    search_depth = st.slider(
        "Search Depth", 
        min_value=2, 
        max_value=8, 
        value=4,
        help="Number of policy chunks to analyze (more = slower but thorough)"
    )
    
    st.markdown("---")
    
    st.markdown("### Architecture")
    st.markdown(f"""
    **Hybrid Approach:**
    - **Embeddings**: all-MiniLM-L6-v2  
      *(Hugging Face - Free)*
    - **LLM**: {model_provider.split(' ')[0]}  
      *({model_provider})*
    - **Vector DB**: FAISS  
      *(343,120 chunks)*
    - **Data**: OPP-115 Corpus  
      *(115 Privacy Policies)*
    
    **Why Hybrid?**
    - HF embeddings are excellent & free
    - GPT-3.5 gives better legal analysis
    - Total cost: ~$2-3 for entire project
    """)

# ============================================
# CUSTOM LLM WRAPPER
# ============================================
class CustomHuggingFaceLLM(LLM):
    repo_id: str
    token: str
    temperature: float = 0.1
    max_new_tokens: int = 512
    
    @property
    def _llm_type(self) -> str:
        return "custom_huggingface"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(
            model=self.repo_id,
            token=self.token
        )
        
        # Using the chat completion API as requested/suggested
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = client.chat.completions.create(
                model=self.repo_id,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Hugging Face API: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"repo_id": self.repo_id}

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_embeddings():
    """Load Hugging Face embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_vectordb(_embeddings):
    """Load FAISS Database"""
    try:
        # FAISS setup reference: https://python.langchain.com/docs/integrations/vectorstores/faiss
        vectorstore = FAISS.load_local(
            "faiss_index_tos_hf", 
            _embeddings,
            allow_dangerous_deserialization=True 
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading database: {e}")
        st.info("Make sure 'faiss_index_tos_hf' folder is in the same directory as app.py")
        return None

# Check if API key is provided
if not api_key:
    st.info("""
    ### API Key Required
    
    Please provide an API key for the selected provider to proceed.
    
    **Why do I need this?**
    - **OpenAI**: Requires a paid key (cheap, high quality).
    - **Hugging Face**: Requires a free token (slower, free).
    """)
    st.stop()

# Load Resources
with st.spinner("Loading AI models..."):
    embeddings = load_embeddings()
    db = load_vectordb(embeddings)
    
    if db is None:
        st.stop()

    # Load LLM based on selection
    if model_provider == "OpenAI (Recommended)":
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=600
        )
    else:
        llm = CustomHuggingFaceLLM(
            repo_id=hf_model_id,
            token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            temperature=0.1,
            max_new_tokens=512
        )

st.success(f"System Ready! Knowledge Base: {db.index.ntotal:,} policy chunks loaded")

# ============================================
# RAG CHAIN
# ============================================
# Hey team, I made this template for you to use in your RAG chain. You can use it as is or modify it to your liking.

template = """You are a consumer rights advocate and legal expert specializing in translating Terms of Service into plain English.

Your task is to analyze privacy policies and answer questions at an 8th-grade reading level while maintaining complete accuracy.

CRITICAL RULES:
1. ONLY use information from the Context below
2. If the answer is not in the Context, respond: "I cannot find this specific information in the provided documents."
3. Never invent, assume, or hallucinate information
4. Always cite which part of the document you're referencing
5. Translate legal jargon into simple, clear language
6. Assess the risk level for users (Low/Medium/High)
7. Explain WHY something matters to the average user

Context from legal documents:
{context}

User Question: {question}

Response Format:
**Answer**: [Your clear, direct answer in plain English]

**Risk Level**: [Low / Medium / High]

**Source Quote**: "[Exact text from the document above]"

**Why This Matters**: [Explain in 1-2 sentences why users should care]
"""

PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# RAG implementation guide: https://python.langchain.com/docs/use_cases/question_answering/
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": search_depth}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ============================================
# MAIN INTERFACE
# ============================================

# We learned how to implement the frontend using Streamlit by following this tutorial: https://docs.streamlit.io/en/stable/ and https://www.youtube.com/watch?v=8W8NQFFbDcU

tabs = st.tabs(["Live Analysis", "Knowledge Base", "Common Concerns", "About", "Resources"])

# TAB 1: Live Analysis
with tabs[0]:
    st.header("Live Policy Analysis")
    st.markdown("Paste a URL or text from a specific privacy policy to audit it.")
    
    # Session state for temporary DB
    if 'temp_db' not in st.session_state:
        st.session_state.temp_db = None
    
    col1, col2 = st.columns([1, 2])
    with col1:
        input_method = st.radio("Input Method", ["URL", "Paste Text"])
    
    policy_text = ""
    
    if input_method == "URL":
        with st.form("url_form"):
            url = st.text_input("Enter Privacy Policy URL", placeholder="https://example.com/privacy")
            submitted = st.form_submit_button("Fetch & Process URL")
            
            if submitted:
                if url:
                    with st.spinner(f"Fetching {url}..."):
                        try:
                            response = requests.get(url, timeout=10)
                            soup = BeautifulSoup(response.content, 'html.parser')
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            policy_text = soup.get_text(separator='\n')
                            st.success("Content fetched successfully!")
                            # Store in session state to persist after reload
                            st.session_state.policy_text = policy_text
                        except Exception as e:
                            st.error(f"Error fetching URL: {e}")
                else:
                    st.warning("Please enter a URL")
    else:
        with st.form("text_form"):
            raw_text = st.text_area("Paste Policy Text", height=200, placeholder="Paste the full text of a privacy policy here...")
            submitted = st.form_submit_button("Process Text")
            if submitted:
                policy_text = raw_text
                st.session_state.policy_text = policy_text

    # Check if we have policy text in session state (from form submission)
    if 'policy_text' in st.session_state and st.session_state.policy_text:
        policy_text = st.session_state.policy_text

    if policy_text and (not st.session_state.temp_db or 'last_policy' not in st.session_state or st.session_state.last_policy != policy_text[:50]):
        with st.spinner("Processing and Indexing..."):
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(policy_text)
                
                # Create temporary FAISS index
                st.session_state.temp_db = FAISS.from_texts(chunks, embeddings)
                st.session_state.last_policy = policy_text[:50] # Marker to avoid re-indexing
                st.success(f"Indexed {len(chunks)} chunks from new policy!")
            except Exception as e:
                st.error(f"Error processing text: {e}")

    st.markdown("---")
    
    if st.session_state.temp_db:
        st.subheader("Ask about this specific policy")
        with st.form("live_q_form"):
            live_question = st.text_input("Question", placeholder="Does this policy allow data selling?", key="live_q")
            submitted = st.form_submit_button("Ask AI")
            
            if submitted and live_question:
                with st.spinner("Analyzing..."):
                    try:
                        # Create a temporary chain
                        live_qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.temp_db.as_retriever(search_kwargs={"k": 4}),
                            chain_type_kwargs={"prompt": PROMPT},
                            return_source_documents=True
                        )
                        
                        result = live_qa_chain.invoke({"query": live_question})
                        
                        st.markdown("### 🤖 Analysis Result")
                        st.markdown(result['result'])
                        
                        with st.expander("📄 View Source Excerpts"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.markdown(f"**Excerpt {i}:**")
                                st.text(doc.page_content)
                                st.markdown("---")
                                
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Process a policy above to start asking questions.")

# TAB 2: Knowledge Base
with tabs[1]:
    st.header("Explore the Privacy Policy Landscape")
    st.markdown("""
    **Aggregated Industry Insights**
    
    This tool analyzes a dataset of **115 different privacy policies** to identify common industry patterns and legal standards.
    
    > **Note:** Since the data is aggregated, answers in this tab reflect **general trends** and cannot be attributed to a specific company.
    > To audit a specific company (like Google or Facebook), please use the **Live Analysis** tab.
    """)
    
    with st.expander("Example Research Questions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Industry Trends:**
            - What is the standard data retention period?
            - How do companies usually handle arbitration?
            - Is it common to sell user data?
            """)
        
        with col2:
            st.markdown("""
            **Legal Standards:**
            - What are the typical GDPR requirements?
            - How do policies define 'personal data'?
            - What are common opt-out mechanisms?
            """)
    
    with st.form("kb_form"):
        user_question = st.text_input(
            "Research the database:", 
            placeholder="e.g., What is the most common reason for account termination?",
            key="main_question"
        )
        submitted = st.form_submit_button("Search Database")
    
    if submitted and user_question:
        with st.spinner("Searching 3,929 policy chunks..."):
            try:
                result = qa_chain.invoke({"query": user_question})
                
                # Display answer
                st.markdown("### AI Analysis")
                st.markdown(result['result'])
                
                # Show source documents
                with st.expander("View Source Documents (Click to Verify)"):
                    st.markdown("These are the actual policy excerpts used to generate the answer:")
                    
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}** (from {doc.metadata.get('source', 'unknown')} set):")
                        st.text_area(
                            f"Policy Excerpt {i}",
                            doc.page_content,
                            height=150,
                            key=f"source_{i}_{user_question}",
                            disabled=True
                        )
                        st.markdown("---")
                
                # Feedback
                st.markdown("---")
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Helpful", key="helpful"):
                        st.success("Thanks!")
                with col2:
                    if st.button("Not helpful", key="not_helpful"):
                        st.info("Try being more specific or rephrasing your question!")
                        
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Check that your API key/token is valid")

# TAB 3: Common Concerns
with tabs[2]:
    st.header("Common Privacy Concerns")
    st.markdown("*Pre-built analysis of frequently asked privacy questions*")
    
    concerns = {
        "Data Selling": "Does the policy allow selling personal data to third parties for profit?",
        "AI Training": "Can the company use my content, photos, or messages to train AI models?",
        "Forced Arbitration": "Am I forced into binding arbitration instead of suing in court?",
        "Data Deletion": "Can I request complete deletion of all my personal data?",
        "Third-Party Sharing": "Do they share my information with advertisers or partners?",
        "Data Retention": "How long do they keep my personal information?",
    }
    
    selected_concern = st.selectbox(
        "Select a concern to analyze:",
        list(concerns.keys()),
        key="concern_select"
    )
    
    if st.button("Analyze this concern", type="primary"):
        query = concerns[selected_concern]
        
        with st.spinner(f"Analyzing: {selected_concern}"):
            try:
                result = qa_chain.invoke({"query": query})
                
                st.markdown(f"### {selected_concern}")
                st.markdown(result['result'])
                
                with st.expander("View Sources"):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.text_area(
                            f"Source {i}",
                            doc.page_content,
                            height=100,
                            key=f"concern_source_{i}_{selected_concern}"
                        )
                        
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 4: About
with tabs[3]:
    st.header("Project Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Policies", ">115")
        st.metric("Policy Chunks", "343,120")
    
    with col2:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
        st.metric("Vector DB", "FAISS")
        
    with col3:
        st.metric("LLM Provider", model_provider.split(' ')[0])
        
    st.markdown("---")
    st.markdown("""
    **ToS RAG Analyzer** is an academic project designed to make privacy policies transparent.
    
    **Features:**
    - **RAG Architecture**: Retrieves real policy text to ground answers.
    - **Dual LLM Support**: Compare OpenAI (Commercial) vs Hugging Face (Open Source).
    - **Live Analysis**: Audit any policy in real-time.
    
    **Created by Group 34** for SFU IAT 360.
    """)
    
    st.subheader("Academic Context")
    st.markdown("""
    **Problem Statement**:  
    91% of users accept Terms of Service without reading them because these documents 
    average 10,000+ words and require college-level reading comprehension.
    
    **Solution**:  
    A RAG system that retrieves relevant policy sections and translates them into 
    plain English using AI.
    """)
    
    st.markdown("---")
    
    st.subheader("Technical Architecture")
    st.markdown("""
    **Hybrid RAG Pipeline**:
```
    User Question 
      → Embedding (HF: all-MiniLM-L6-v2)
      → FAISS Search (Top 4 chunks)
      → GPT-3.5 Analysis
      → Plain English Answer
```
    
    **Why This Architecture?**
    - **Hugging Face Embeddings**: Free, open-source, excellent quality
    - **OpenAI LLM**: Superior legal text comprehension
    - **FAISS**: Fast vector similarity search
    - **RAG**: Prevents hallucinations, provides citations
    """)
    
    st.markdown("---")
    
    st.subheader("Ethical Considerations")
    
    st.markdown("""
    This project acknowledges several inherent biases in the underlying data and models. The dataset primarily consists of privacy policies from major US-based technology companies, which may not reflect global legal standards or the practices of smaller organizations. Additionally, the data was collected around 2016-2019, meaning it may not fully capture recent regulatory changes like GDPR or CCPA in all instances.
    
    To mitigate these risks, the application includes clear disclaimers that the AI-generated content is not legal advice. We explicitly encourage users to verify findings against the original policy text, which is always provided alongside the analysis. The goal is to augment human understanding, not replace professional legal counsel.
    """)

# TAB 5: Resources
with tabs[4]:
    st.header("Project Resources & References")
    
    st.markdown("""
    Here is a collection of resources, libraries, and references used in the development of this application.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Development & Tutorials")
        st.markdown("""
        - **Streamlit Documentation**:  
          [docs.streamlit.io](https://docs.streamlit.io/en/stable/)
        - **Streamlit Tutorial (YouTube)**:  
          [Building a Data App in Streamlit](https://www.youtube.com/watch?v=8W8NQFFbDcU)
        - **Pydantic Fix Discussion**:  
          [GitHub Discussion #6766](https://github.com/pydantic/pydantic/discussions/6766)  
          *(Solution for Pydantic v1/v2 compatibility issues)*
        """)
        
    with col2:
        st.subheader("APIs & Tools")
        st.markdown("""
        - **OpenAI API Keys**:  
          [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        - **Hugging Face Tokens**:  
          [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        - **LangChain**:  
          [python.langchain.com](https://python.langchain.com/docs/get_started/introduction)
        - **FAISS (Facebook AI Similarity Search)**:  
          [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
        - **Real Python BeautifulSoup Tutorial**:  
          [realpython.com](https://realpython.com/beautiful-soup-web-scraper-python/)
        - **LangChain RAG Tutorial**:  
          [python.langchain.com](https://python.langchain.com/docs/use_cases/question_answering/)
        - **FAISS Vector Store Guide**:  
          [python.langchain.com](https://python.langchain.com/docs/integrations/vectorstores/faiss)
        """)
    
    st.markdown("---")
    st.caption("These resources were instrumental in building the ToS RAG Analyzer.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>This tool is for educational purposes only</strong></p>
    <p>AI-generated interpretations are NOT legal advice.</p>
    <p style='margin-top: 1.5rem;'>
        <strong>IAT 360 Final Project</strong> • Simon Fraser University<br>
        Hybrid RAG Architecture: HuggingFace + OpenAI
    </p>
</div>
""", unsafe_allow_html=True)