from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def inspect_content():
    print("Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        db = FAISS.load_local(
            "faiss_index_tos_hf", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        docstore = db.docstore._dict
        print(f"Docstore size: {len(docstore)}")
        
        # Print the content of the first 5 documents to see if company names are present
        sample_count = 0
        for key, doc in docstore.items():
            if sample_count < 5:
                print(f"\n--- Document {sample_count+1} ---")
                print(f"Metadata: {doc.metadata}")
                print(f"Content Start: {doc.page_content[:200]}...") # Print first 200 chars
                sample_count += 1
            else:
                break
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_content()
