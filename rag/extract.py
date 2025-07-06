import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.schema import Document
import chromadb

class RAGPipeline:
    def __init__(self, model_name="mistral:7b-instruct", persist_directory="./chroma_db"):
        """
        Initialize RAG Pipeline
        
        Args:
            model_name: Ollama model name
            persist_directory: Directory to store ChromaDB data
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup all RAG components"""
        print("Setting up RAG components...")
        
        # Initialize Ollama LLM
        print("Initializing Ollama LLM...")
        self.llm = Ollama(model=self.model_name)
        
        # Initialize embeddings (using local HuggingFace model)
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize or load existing ChromaDB
        print("Setting up ChromaDB...")
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Loaded existing ChromaDB")
        else:
            # Will be created when documents are added
            print("ChromaDB will be created when documents are added")
        
        print("RAG components initialized successfully!")
    
    def load_document(self, file_path):
        """
        Load a single document
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        print(f"Loading document: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Determine file type and use appropriate loader
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            # Try to load as text file
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s)")
        return documents
    

    
    def process_documents(self, documents):
        """
        Process documents: chunk, embed, and store in ChromaDB
        
        Args:
            documents: List of Document objects
        """
        print("Processing documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        
        # Create or update ChromaDB
        if self.vectorstore is None:
            print("Creating new ChromaDB...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            print("Adding to existing ChromaDB...")
            self.vectorstore.add_documents(chunks)
        
        # Persist the database
        self.vectorstore.persist()
        print("Documents processed and stored in ChromaDB")
        
        # Setup retriever and QA chain
        self._setup_retrieval_chain()
    
    def _create_custom_prompt(self):
        """Create custom prompt template with strict document-only instructions"""
        from langchain.prompts import PromptTemplate
        
        prompt_template = """You are a document analysis assistant. Follow these instructions STRICTLY:

RULES:
1. Answer ONLY based on the provided context from the documents
2. DO NOT add any information from your general knowledge
3. DO NOT make assumptions or inferences beyond what is explicitly stated
4. If the answer is not present in the provided context, respond EXACTLY with: "This information is not available in the provided document."
5. When answering, reference which part of the document your answer comes from

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

ANSWER (following the rules above):"""
        
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
    def _setup_retrieval_chain(self):
        """Setup retrieval chain for QA"""
        print("Setting up retrieval chain...")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Return top 4 relevant chunks
        )
        
        # Create custom prompt with strict instructions
        custom_prompt = self._create_custom_prompt()
        
        # Create QA chain with custom prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
            verbose=True
        )
        
        print("Retrieval chain setup complete!")
    
    def ask_question(self, question):
        """
        Ask a question about the documents
        
        Args:
            question: Question string
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("No documents loaded. Please load documents first.")
        
        print(f"\nQuestion: {question}")
        print("Searching for relevant information...")
        
        # Get answer
        result = self.qa_chain({"query": question})
        
        print(f"\nAnswer: {result['result']}")
        print(f"\nSources used: {len(result['source_documents'])} chunks")
        
        return result
    
    def chat_loop(self):
        """Start an interactive chat loop"""
        print("\n" + "="*50)
        print("RAG CHATBOT - Ask questions about your documents!")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                result = self.ask_question(question)
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    """Main function to run the RAG pipeline"""
    print("Starting RAG Pipeline...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Load user's document
    file_path = input("Enter path to your document (PDF, TXT, etc.): ").strip()
    
    try:
        documents = rag.load_document(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please check the file path and try again.")
        return
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return
    
    # Process documents
    rag.process_documents(documents)
    
    # Start chat loop
    rag.chat_loop()

if __name__ == "__main__":
    main()