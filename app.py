import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="ToS Analyzer - Hybrid Edition",
    page_icon="⚖️",
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    <span class='emoji-header'>⚖️</span> 
    <span class='main-header'>ToS Analyzer</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem;'>Understand Terms of Service in Plain English</p>
    <p style='color: #666;'>Hybrid Architecture: Hugging Face (Embeddings) + OpenAI (Reasoning)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Important Disclaimer</strong><br>
    This tool provides AI-generated interpretations. It is <strong>NOT legal advice</strong>. 
    Always read the full Terms of Service and consult a qualified attorney for legal decisions.
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("🔑 Authentication")
    
    st.markdown("""
    **Get your OpenAI API Key:**
    1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    2. Create new secret key
    3. Paste below
    
    *New accounts get $5 free credit*
    """)
    
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ Connected to OpenAI!")
    else:
        st.warning("⚠️ Waiting for API Key...")
    
    st.markdown("---")
    
    st.header("⚙️ Settings")
    search_depth = st.slider(
        "Search Depth", 
        min_value=2, 
        max_value=8, 
        value=4,
        help="Number of policy chunks to analyze (more = slower but thorough)"
    )
    
    st.markdown("---")
    
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    **Hybrid Approach:**
    - 🔍 **Embeddings**: all-MiniLM-L6-v2  
      *(Hugging Face - Free)*
    - 🧠 **LLM**: GPT-3.5-Turbo  
      *(OpenAI - ~$0.002/query)*
    - 💾 **Vector DB**: FAISS  
      *(3,929 chunks)*
    - 📚 **Data**: OPP-115 Corpus  
      *(115 real policies)*
    
    **Why Hybrid?**
    - ✅ HF embeddings are excellent & free
    - ✅ GPT-3.5 gives better legal analysis
    - ✅ Total cost: ~$2-3 for entire project
    """)
    
    with st.expander("💰 Cost Breakdown"):
        st.markdown("""
        **Per Query Estimate:**
        - Embeddings: $0 (Hugging Face)
        - Input tokens (~800): $0.0008
        - Output tokens (~300): $0.0006
        - **Total: ~$0.0014 per query**
        
        **For 100 test queries: ~$0.14**
        """)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_embeddings():
    """Load Hugging Face embeddings (FREE & OPEN SOURCE)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_vectordb(_embeddings):
    """Load FAISS Database"""
    try:
        vectorstore = FAISS.load_local(
            "faiss_index_tos_hf", 
            _embeddings,
            allow_dangerous_deserialization=True  # FIXED: Added this parameter
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        st.info("💡 Make sure 'faiss_index_tos_hf' folder is in the same directory as app.py")
        return None

# Check if API key is provided
if not api_key:
    st.info("""
    ### 🔑 API Key Required
    
    This app uses OpenAI's GPT-3.5-turbo for high-quality legal analysis.
    
    **Setup Steps:**
    1. Go to https://platform.openai.com/api-keys
    2. Create a new API key (free $5 credit for new accounts)
    3. Paste it in the sidebar
    
    **Why OpenAI instead of free models?**
    - Better accuracy for legal interpretation
    - Faster response times
    - More reliable for demonstrations
    - Total project cost: ~$2-3
    """)
    st.stop()

# Load Resources
with st.spinner("🔄 Loading AI models..."):
    embeddings = load_embeddings()
    db = load_vectordb(embeddings)
    
    if db is None:
        st.stop()

    # Load OpenAI LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,  # Low temperature for factual responses
        max_tokens=600    # Enough for detailed answers
    )

st.success(f"✅ System Ready! Knowledge Base: {db.index.ntotal:,} policy chunks loaded")

# ============================================
# RAG CHAIN
# ============================================
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

tabs = st.tabs(["💬 Ask Questions", "🧪 Live Test", "🔍 Common Concerns", "📊 About"])

# TAB 1: Question Answering
with tabs[0]:
    st.header("Ask About Privacy Policies")
    
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Collection:**
            - What personal data do they collect?
            - Do they track my location?
            - Can they access my contacts?
            
            **Data Usage:**
            - Can they sell my data?
            - Do they use my content for AI training?
            - Do they share data with advertisers?
            """)
        
        with col2:
            st.markdown("""
            **User Rights:**
            - Can I delete all my data?
            - Can I opt out of data collection?
            - Do I have GDPR rights?
            
            **Legal Terms:**
            - Am I forced into arbitration?
            - Can they change terms without notice?
            - What happens if they get hacked?
            """)
    
    user_question = st.text_input(
        "What do you want to know?", 
        placeholder="e.g., Can this company use my photos for advertising?",
        key="main_question"
    )
    
    if user_question:
        with st.spinner("🔍 Searching 3,929 policy chunks..."):
            try:
                result = qa_chain.invoke({"query": user_question})
                
                # Display answer
                st.markdown("### 🤖 AI Analysis")
                st.markdown(result['result'])
                
                # Show source documents
                with st.expander("📄 View Source Documents (Click to Verify)"):
                    st.markdown("*These are the actual policy excerpts used to generate the answer:*")
                    
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
                    if st.button("👍 Helpful", key="helpful"):
                        st.success("Thanks!")
                with col2:
                    if st.button("👎 Not helpful", key="not_helpful"):
                        st.info("Try being more specific or rephrasing your question!")
                        
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Check that your OpenAI API key is valid and has credits")

# TAB 2: Live Test
with tabs[1]:
    st.header("🧪 Live Policy Analysis")
    st.markdown("Test the model on *new* data by providing a URL or pasting text.")
    
    # Session state for temporary DB
    if 'temp_db' not in st.session_state:
        st.session_state.temp_db = None
    
    col1, col2 = st.columns([1, 2])
    with col1:
        input_method = st.radio("Input Method", ["URL", "Paste Text"])
    
    policy_text = ""
    
    if input_method == "URL":
        url = st.text_input("Enter Privacy Policy URL", placeholder="https://example.com/privacy")
        if st.button("Fetch & Process URL"):
            if url:
                with st.spinner(f"Fetching {url}..."):
                    try:
                        response = requests.get(url, timeout=10)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        policy_text = soup.get_text(separator='\n')
                        st.success("✅ Content fetched successfully!")
                    except Exception as e:
                        st.error(f"❌ Error fetching URL: {e}")
            else:
                st.warning("Please enter a URL")
    else:
        policy_text = st.text_area("Paste Policy Text", height=200, placeholder="Paste the full text of a privacy policy here...")
        if st.button("Process Text"):
            pass # Trigger processing below

    if policy_text:
        with st.spinner("🧠 Processing and Indexing..."):
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(policy_text)
                
                # Create temporary FAISS index
                st.session_state.temp_db = FAISS.from_texts(chunks, embeddings)
                st.success(f"✅ Indexed {len(chunks)} chunks from new policy!")
            except Exception as e:
                st.error(f"❌ Error processing text: {e}")

    st.markdown("---")
    
    if st.session_state.temp_db:
        st.subheader("Ask about this specific policy")
        live_question = st.text_input("Question", placeholder="Does this policy allow data selling?", key="live_q")
        
        if live_question:
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
        st.info("👆 Process a policy above to start asking questions.")

# TAB 3: Common Concerns
with tabs[2]:
    st.header("🔍 Common Privacy Concerns")
    st.markdown("*Pre-built analysis of frequently asked privacy questions*")
    
    concerns = {
        "🔴 Data Selling": "Does the policy allow selling personal data to third parties for profit?",
        "🤖 AI Training": "Can the company use my content, photos, or messages to train AI models?",
        "⚖️ Forced Arbitration": "Am I forced into binding arbitration instead of suing in court?",
        "🗑️ Data Deletion": "Can I request complete deletion of all my personal data?",
        "📊 Third-Party Sharing": "Do they share my information with advertisers or partners?",
        "⏰ Data Retention": "How long do they keep my personal information?",
    }
    
    selected_concern = st.selectbox(
        "Select a concern to analyze:",
        list(concerns.keys()),
        key="concern_select"
    )
    
    if st.button("🔍 Analyze This Concern", type="primary"):
        query = concerns[selected_concern]
        
        with st.spinner(f"Analyzing: {selected_concern}"):
            try:
                result = qa_chain.invoke({"query": query})
                
                st.markdown(f"### {selected_concern}")
                st.markdown(result['result'])
                
                with st.expander("📄 View Sources"):
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
    st.header("📊 Project Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Policies", "115")
        st.metric("Policy Chunks", "3,929")
    
    with col2:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
        st.metric("LLM Model", "GPT-3.5-turbo")
    
    with col3:
        st.metric("Vector Database", "FAISS")
        st.metric("Data Source", "OPP-115")
    
    st.markdown("---")
    
    st.subheader("🎓 Academic Context")
    st.markdown("""
    **Course**: IAT 360 - Designing the Digital Future  
    **Institution**: Simon Fraser University  
    **Project Type**: Option 2 - Deploy Existing Model (RAG)
    
    **Problem Statement**:  
    91% of users accept Terms of Service without reading them because these documents 
    average 10,000+ words and require college-level reading comprehension.
    
    **Solution**:  
    A RAG system that retrieves relevant policy sections and translates them into 
    plain English using AI.
    """)
    
    st.markdown("---")
    
    st.subheader("🏗️ Technical Architecture")
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
    
    st.subheader("⚖️ Ethical Considerations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Known Biases**:
        - 🇺🇸 US-centric legal frameworks
        - 🗓️ Dataset from 2019 (may be outdated)
        - 🌐 English language only
        - 🏢 Large companies overrepresented
        """)
    
    with col2:
        st.markdown("""
        **Mitigations**:
        - ✅ Clear disclaimers throughout UI
        - ✅ Source citations for verification
        - ✅ "Not legal advice" warnings
        - ✅ Encourages reading full policies
        """)
    
    st.markdown("---")
    
    st.subheader("📜 Licenses & Attribution")
    st.markdown("""
    **Models**:
    - GPT-3.5-turbo: OpenAI (Commercial license)
    - all-MiniLM-L6-v2: Apache 2.0
    
    **Data**:
    - OPP-115 Corpus: Wilson et al. (2016) - Academic use permitted
    
    **Code**:
    - MIT License (educational project)
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>⚠️ This tool is for educational purposes only</strong></p>
    <p>AI-generated interpretations are NOT legal advice.</p>
    <p style='margin-top: 1.5rem;'>
        <strong>IAT 360 Final Project</strong> • Simon Fraser University<br>
        Hybrid RAG Architecture: HuggingFace + OpenAI
    </p>
</div>
""", unsafe_allow_html=True)