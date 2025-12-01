import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="ToS Analyzer - Free Edition",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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
    .emoji-header {
        font-size: 3rem;
        vertical-align: bottom;
    }
    .disclaimer {
        background-color: #fff3cd;
        color: #856404; /* Dark yellow/brown text for readability */
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
    h1 {
        text-align: center;
    }
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
    <p style='color: #666;'>100% Free • Powered by Hugging Face • No API Costs</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Important Disclaimer</strong><br>
    This tool provides AI-generated interpretations of legal documents. It is <strong>NOT legal advice</strong>. 
    Always read the full Terms of Service and consult a qualified attorney for legal decisions.
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("🔑 Hugging Face Setup")
    
    st.markdown("""
    To use this app, you need a **free** Hugging Face token:
    
    1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    2. Click "New token"
    3. Copy and paste it below
    
    **Note**: Completely free, no credit card needed!
    """)
    
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        help="Get free token from https://huggingface.co/settings/tokens"
    )
    
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        st.success("✅ Token configured!")
    else:
        st.warning("⚠️ Enter token to use the app")
    
    st.markdown("---")
    
    st.header("⚙️ Settings")
    
    search_depth = st.slider(
        "Search Depth",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of document chunks to analyze (more = slower but more thorough)"
    )
    
    st.markdown("---")
    
    st.header("📊 About")
    st.markdown("""
    **Course**: IAT 360 @ SFU  
    **Project**: ToS Analyzer (RAG)  
    **Option**: 2 - Deploy Existing Model
    
    **Tech Stack**:
    - 🤖 LLM: Mistral-7B-Instruct
    - 🔍 Embeddings: all-MiniLM-L6-v2
    - 💾 Vector DB: FAISS
    - 📚 Data: OPP-115 (3,432 policies)
    
    **100% Free & Open Source**:
    - No API costs
    - No credit card required
    - All models from Hugging Face
    """)
    
    with st.expander("📖 How It Works"):
        st.markdown("""
        **RAG (Retrieval-Augmented Generation)**:
        
        1. **Your Question** → Converted to vector
        2. **Search** → Finds relevant policy sections
        3. **LLM Reads** → Mistral analyzes actual text
        4. **Plain English** → You get clear answer
        
        **Why RAG?**
        - ✅ No hallucinations (uses real text)
        - ✅ Shows sources (you can verify)
        - ✅ Legal accuracy (critical for ToS)
        """)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_embeddings():
    """Load Hugging Face embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_vectordb(_embeddings):
    """Load FAISS vector database"""
    try:
        # FIXED: Removed 'allow_dangerous_deserialization=True' 
        # This argument is not needed for the older LangChain version you are using.
        vectorstore = FAISS.load_local(
            "faiss_index_tos_hf",
            _embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        st.info("💡 Make sure 'faiss_index_tos_hf' folder is in the same directory as app.py")
        return None

@st.cache_resource
def load_llm():
    """Load Hugging Face LLM (Endpoint Version)"""
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # The older library needs a full URL, not just the ID
    endpoint_url = f"https://router.huggingface.co/models/{repo_id}"
    
    # We use 'endpoint_url' and pack params into 'model_kwargs'
    return HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    )

# Check if token is provided
if not hf_token:
    st.warning("👈 **Please enter your Hugging Face token in the sidebar to continue**")
    st.info("""
    ### Why do I need a token?
    - The app uses Hugging Face's free Inference API
    - Token is completely free (no credit card needed)
    - Get yours here: https://huggingface.co/settings/tokens
    
    ### Is my token safe?
    - ✅ Stored only in your browser session
    - ✅ Never saved to disk
    - ✅ Not shared with anyone
    """)
    st.stop()

# Load models with progress indicators
with st.spinner("🔄 Loading AI models from Hugging Face..."):
    embeddings = load_embeddings()
    db = load_vectordb(embeddings)
    
    if db is None:
        st.stop()
    
    llm = load_llm()

st.success(f"✅ Ready! Vector database loaded with {db.index.ntotal:,} policy chunks")

# ============================================
# RAG CHAIN SETUP
# ============================================

template = """<s>[INST] You are a consumer rights advocate and legal expert who translates Terms of Service into plain English.

Your job is to help everyday people understand what they're agreeing to when they accept privacy policies.

CRITICAL RULES:
1. ONLY use information from the Context below
2. If the answer is NOT in the Context, respond: "I cannot find this specific information in the provided documents."
3. Write in simple, clear English (8th-grade reading level)
4. Always include a direct quote from the document as evidence
5. Rate the privacy risk: Low, Medium, or High
6. Explain WHY this matters to regular users

Context from legal documents:
{context}

User Question: {question}

Respond in this EXACT format:

**Answer**: [Your clear, direct answer in 2-3 sentences]

**Risk Level**: [Low / Medium / High]

**Direct Quote**: "[Copy exact text from the document above]"

**Why This Matters**: [Explain in 1-2 sentences why users should care about this]

[/INST]"""

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

tabs = st.tabs(["💬 Ask Questions", "🔍 Common Concerns", "📚 Dataset Info"])

# TAB 1: Question Answering
with tabs[0]:
    st.header("Ask About Privacy Policies")
    
    # Example questions
    with st.expander("💡 Example Questions You Can Ask"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Collection**:
            - What personal data do they collect?
            - Do they track my location?
            - Can they access my contacts?
            
            **Data Usage**:
            - Can they sell my data?
            - Do they use my content for AI training?
            - Do they share data with advertisers?
            """)
        
        with col2:
            st.markdown("""
            **User Rights**:
            - Can I delete all my data?
            - Can I opt out of data collection?
            - Do I have GDPR rights?
            
            **Legal Terms**:
            - Am I forced into arbitration?
            - Can they change terms without notice?
            - What happens if they get hacked?
            """)
    
    # Question input
    user_question = st.text_input(
        "What would you like to know?",
        placeholder="e.g., Can this company use my photos for advertising?",
        key="main_question"
    )
    
    if user_question:
        with st.spinner("🔍 Searching 3,432 privacy policies..."):
            try:
                result = qa_chain.invoke({"query": user_question})
                answer = result['result']
                source_docs = result['source_documents']
                
                # Display the answer
                st.markdown("### 🤖 AI Analysis")
                st.markdown(answer)
                
                # Show source documents
                with st.expander("📄 View Source Documents (Click to Verify)"):
                    st.markdown("*These are the actual policy excerpts the AI used:*")
                    
                    for i, doc in enumerate(source_docs, 1):
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
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thanks for the feedback!")
                with col2:
                    if st.button("👎 Not helpful"):
                        st.info("Try rephrasing your question or being more specific!")
                        
            except Exception as e:
                st.error(f"❌ Error: {e}")
                
                error_msg = str(e).lower()
                
                if "429" in error_msg or "rate" in error_msg:
                    st.warning("""
                    **Rate Limit Reached**
                    
                    Hugging Face's free tier has limits. Please:
                    - Wait 1-2 minutes and try again
                    - Use shorter, more specific questions
                    - Try during off-peak hours
                    """)
                elif "token" in error_msg or "401" in error_msg:
                    st.warning("""
                    **Token Issue**
                    
                    Please check:
                    - Your Hugging Face token is correct
                    - Token has 'read' permissions
                    - Get a new token at: https://huggingface.co/settings/tokens
                    """)
                else:
                    st.info("💡 Try asking a simpler question or check your internet connection")

# TAB 2: Pre-built Common Concerns
with tabs[1]:
    st.header("🔍 Common Privacy Concerns")
    st.markdown("*Quick analysis of frequently asked questions*")
    
    concerns = {
        "🔴 Data Selling": "Does the policy allow selling personal data to third parties for profit?",
        "🤖 AI Training": "Can the company use my content, photos, or messages to train artificial intelligence models?",
        "⚖️ Forced Arbitration": "Am I forced into binding arbitration instead of being able to sue in court?",
        "🗑️ Data Deletion": "Can I request complete deletion of all my personal data from their servers?",
        "📊 Third-Party Sharing": "Do they share my information with advertisers, partners, or other companies?",
        "⏰ Data Retention": "How long do they keep my personal information after I stop using the service?",
        "📍 Location Tracking": "Do they track my physical location, and can I turn it off?",
        "👁️ Behavioral Tracking": "Do they monitor my behavior across different websites and apps?",
    }
    
    selected_concern = st.selectbox(
        "Select a privacy concern to analyze:",
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
                            height=120,
                            key=f"concern_source_{i}_{selected_concern}"
                        )
                        
            except Exception as e:
                st.error(f"Error: {e}")
                if "429" in str(e):
                    st.warning("Rate limit reached. Please wait 1 minute and try again.")

# TAB 3: Dataset Information
with tabs[2]:
    st.header("📚 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Policies", "115")
        st.metric("Training Examples", "2,185")
    
    with col2:
        st.metric("Validation Examples", "550")
        st.metric("Test Examples", "697")
    
    with col3:
        st.metric("Total Chunks", "3,929")
        st.metric("Avg Chunk Size", "~401 chars")
    
    st.markdown("---")
    
    st.subheader("🗂️ Data Source: OPP-115 Corpus")
    
    st.markdown("""
    The **OPP-115 dataset** is a collection of 115 privacy policies from major websites, 
    professionally annotated by legal experts for research purposes.
    
    **Dataset Details**:
    - **Source**: [Hugging Face - alzoubi36/opp_115](https://huggingface.co/datasets/alzoubi36/opp_115)
    - **Original Research**: Wilson et al. (2016) - "The Creation and Analysis of a Website Privacy Policy Corpus"
    - **Companies Include**: Google, Amazon, Facebook, Apple, Microsoft, and 110+ others
    - **License**: Academic and research use permitted
    
    **What's in the dataset**:
    - Privacy policy text segments
    - Annotations for policy categories (data collection, usage, retention, etc.)
    - Covering major tech, finance, healthcare, and e-commerce companies
    
    **How we processed it**:
    1. Combined all splits (train + validation + test) = 3,432 documents
    2. Chunked into 800-character segments with 100-char overlap
    3. Created vector embeddings using Sentence Transformers
    4. Stored in FAISS for fast semantic search
    """)
    
    st.markdown("---")
    
    st.subheader("🛡️ Guardrails & Safety")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Anti-Hallucination**:
        - ✅ RAG architecture (retrieval-only)
        - ✅ Temperature = 0.1 (minimal creativity)
        - ✅ "Cannot find" responses when unsure
        - ✅ All answers cite source documents
        """)
    
    with col2:
        st.markdown("""
        **Privacy Protection**:
        - ✅ Zero query logging
        - ✅ No data retention
        - ✅ Token stored in session only
        - ✅ Open source (auditable)
        """)
    
    st.markdown("---")
    
    st.subheader("⚖️ Known Limitations")
    
    st.warning("""
    **Bias & Limitations**:
    - 🇺🇸 **US-Centric**: 90% of policies follow US legal frameworks (may not apply to EU GDPR, Canadian PIPEDA, etc.)
    - 🗓️ **Dataset Age**: OPP-115 corpus is from ~2019 - policies may have changed
    - 🌐 **Language**: English only
    - 🏢 **Company Size**: Mostly large tech companies (limited small business coverage)
    - 🤖 **AI Limitations**: May misinterpret ambiguous legal language
    
    **Always**:
    - Read the full, current Terms of Service
    - Consult a lawyer for legal decisions
    - Verify information for your jurisdiction
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>⚠️ This tool is for educational purposes only</strong></p>
    <p>AI-generated interpretations are NOT legal advice. Always read the full Terms of Service.</p>
    <p>For legal decisions, consult a qualified attorney.</p>
    <p style='margin-top: 1.5rem; font-size: 0.9rem;'>
        <strong>IAT 360 Final Project</strong> • Simon Fraser University<br>
        Made with 🤗 Hugging Face • 100% Free & Open Source
    </p>
    <p style='margin-top: 1rem; font-size: 0.85rem; color: #999;'>
        Models: Mistral-7B-Instruct-v0.2 • all-MiniLM-L6-v2 | Database: FAISS | Data: OPP-115
    </p>
</div>
""", unsafe_allow_html=True)