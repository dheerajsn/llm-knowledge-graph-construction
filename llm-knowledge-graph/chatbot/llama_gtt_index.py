import fitz  # PyMuPDF
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from PIL import Image
import pytesseract
import io
import os
from pathlib import Path

class GTTIndexCreator:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
    def extract_pdf_with_images(self, pdf_path):
        """Extract text and images from GTT manual PDF"""
        doc = fitz.open(pdf_path)
        documents = []
        
        print(f"Processing {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Get page as image for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # Higher resolution
            img_data = pix.tobytes("png")
            page_image = Image.open(io.BytesIO(img_data))
            
            # Extract text from entire page image using OCR
            try:
                page_ocr_text = pytesseract.image_to_string(page_image, config='--psm 6')
            except:
                page_ocr_text = ""
            
            # Combine extracted text with OCR text
            combined_text = f"Page {page_num + 1} - GTT Documentation\n\n"
            
            if text.strip():
                combined_text += f"Text Content:\n{text}\n\n"
            
            if page_ocr_text.strip() and len(page_ocr_text.strip()) > 50:
                combined_text += f"Visual Content (OCR):\n{page_ocr_text}\n"
            
            documents.append(Document(
                text=combined_text,
                metadata={
                    "page": page_num + 1,
                    "source": "GTT_manual",
                    "doc_type": "documentation",
                    "file_name": Path(pdf_path).name
                }
            ))
            
            print(f"Processed page {page_num + 1}")
        
        doc.close()
        return documents
    
    def create_vector_index(self, pdf_path, persist_dir="./gtt_index_storage"):
        """Create and persist vector index"""
        
        # Extract documents
        documents = self.extract_pdf_with_images(pdf_path)
        
        # Configure chunking
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separator=" "
        )
        
        print("Creating vector index...")
        
        # Create vector index
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
            embed_model=self.embed_model,
            show_progress=True
        )
        
        # Persist index
        print(f"Saving index to {persist_dir}...")
        index.storage_context.persist(persist_dir=persist_dir)
        
        print("Index created and saved successfully!")
        return index

def main():
    # Configuration
    OPENAI_API_KEY = "your-openai-api-key-here"
    PDF_PATH = "path/to/gtt_manual.pdf"
    PERSIST_DIR = "./gtt_index_storage"
    
    # Create index
    creator = GTTIndexCreator(OPENAI_API_KEY)
    index = creator.create_vector_index(PDF_PATH, PERSIST_DIR)
    
    print("✅ GTT Documentation index created successfully!")

if __name__ == "__main__":
    main()