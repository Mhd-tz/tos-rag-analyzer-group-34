# Save this as check_files.py
import os
import pickle

print("🔍 Checking database files...\n")

# Check folder exists
if not os.path.exists("faiss_index_tos_hf"):
    print("❌ Folder 'faiss_index_tos_hf' NOT FOUND")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files here: {os.listdir('.')}")
    exit(1)

print("✅ Folder exists")

# Check files
faiss_file = "faiss_index_tos_hf/index.faiss"
pkl_file = "faiss_index_tos_hf/index.pkl"

for file in [faiss_file, pkl_file]:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024*1024)
        print(f"✅ {file}: {size:.2f} MB")
        
        if size < 0.1:
            print(f"   ⚠️  WARNING: File is suspiciously small!")
    else:
        print(f"❌ {file}: NOT FOUND")

# Try to read the pickle file
print("\n🔍 Testing pickle file...")
try:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        print(f"✅ Pickle loads successfully")
        print(f"   Type: {type(data)}")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())[:5]}")
except Exception as e:
    print(f"❌ Pickle ERROR: {e}")
    print("   → Database is corrupted, needs rebuild")

print("\n" + "="*60)