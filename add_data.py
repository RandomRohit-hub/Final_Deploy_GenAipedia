# add_data.py

from dotenv import load_dotenv
import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# ----------------- LOAD .ENV -----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# ----------------- PINECONE INIT -----------------
index_name = "genativeai-encyclopedia"  # must match existing index
pc = Pinecone(api_key=PINECONE_API_KEY)

print("Connecting to Pinecone index:", index_name)


# ----------------- EMBEDDINGS (CPU SAFE) -----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# ----------------- LOAD EXISTING VECTOR STORE -----------------
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

print("Connected successfully.")


print("Connected successfully.")


# ----------------- TEXT SPLITTER (IMPORTANT FOR LARGE TEXT) -----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# ----------------- ADD NEW KNOWLEDGE SECTIONS -----------------

# Profiles (short data)
profile_docs = [
    Document(
        page_content="Siddhant is a full-stack web developer known for strong discipline, teamwork, and delivering high-quality solutions in real projects.",
        metadata={"source": "Profile_Siddhant"}
    ),
    Document(
        page_content="Srijan is an ML developer recognized for persistence, strong problem-solving skills, and collaborative project leadership.",
        metadata={"source": "Profile_Srijan"}
    ),
    Document(
        page_content="Rohit is an AI/ML enthusiast with hands-on experience in training, optimizing, and deploying machine learning models.",
        metadata={"source": "Profile_Rohit"}
    )
]


# Large Educational Knowledge Base (already expanded earlier)
ai_knowledge = ai_knowledge = """
======================
ARTIFICIAL INTELLIGENCE (AI)
======================

Artificial Intelligence refers to the development of computer systems capable of performing tasks that would normally require human intelligence. These tasks include reasoning, understanding language, recognizing patterns, learning from examples, making decisions, and solving problems. AI does not mean making a machine "human," but rather enabling it to perform cognitive tasks that humans perform.

AI works by combining large amounts of data with algorithms that can learn from patterns within that data. The more data the AI system is trained on, the better it becomes at recognizing relationships and making predictions. For example, AI can identify whether an image contains a dog or a cat by analyzing millions of labeled examples.

AI is used in:
- Health (diagnosing diseases)
- Finance (fraud detection)
- Transportation (self-driving cars)
- Entertainment (recommendation systems)
- Education (personalized tutors)
- Business operations (chatbots and automation)

AI continues to evolve as models become larger, more accurate, and more capable of reasoning rather than just memorization.

======================
MACHINE LEARNING (ML)
======================

Machine Learning is a subset of AI focused on enabling computers to learn patterns and relationships from data without being explicitly programmed. Instead of writing instructions for every task, we provide data and allow the algorithm to learn patterns and make decisions.

Machine Learning works through three major steps:
1) The model receives training data.
2) It identifies patterns and relationships.
3) It uses those patterns to make predictions on new, unseen data.

For example, in email spam detection, ML algorithms learn which words, phrases, and patterns frequently occur in spam messages and then flag new messages with similar patterns as spam.

Machine Learning Types:
1. Supervised Learning: The model is trained with labeled data where input and output are known. Used for classification (spam / not spam) and regression (predicting house prices).
2. Unsupervised Learning: The model learns patterns without labeled data. Used for clustering, such as segmenting customers based on behavior.
3. Reinforcement Learning: The model learns by performing actions and receiving rewards or penalties. Used in robotics and game-playing systems like AlphaGo.

ML success depends on:
- Quality of training data
- Proper feature selection
- Effective optimization techniques
- Correct evaluation metrics

======================
DEEP LEARNING (DL)
======================

Deep Learning is an advanced form of machine learning that uses neural networks with multiple layers. These layers automatically extract and learn features from data rather than requiring manual feature engineering.

Deep Learning is especially powerful when dealing with unstructured data like images, video, speech, and natural language. Traditional machine learning struggles with these because the patterns are complex, non-linear, and high-dimensional.

Neural networks learn by adjusting internal weights using a process called backpropagation. Each layer transforms the data, and deeper layers capture more abstract patterns. For example:
- First layers in an image model detect edges
- Middle layers detect shapes
- Final layers detect full objects like faces or vehicles

Deep Learning enables:
- Face recognition systems
- Speech assistants like Alexa and Siri
- Self-driving car vision systems
- Real-time translation systems

However, DL requires:
- Large amounts of labeled data
- High computational power (often GPUs)
- Careful tuning to avoid overfitting

======================
NATURAL LANGUAGE PROCESSING (NLP)
======================

NLP focuses on enabling machines to understand and generate human language. Language is complex due to grammar rules, cultural meanings, tone, sarcasm, and ambiguity.

Earlier NLP systems relied on hand-crafted rules, but modern NLP uses deep learning and transformers.

Key NLP tasks:
- Sentiment analysis: Detecting emotions in text
- Text classification: Categorizing messages, emails, etc.
- Named Entity Recognition (NER): Identifying names, places, dates
- Language translation
- Summarization of documents
- Question answering and chatbots

Transformers revolutionized NLP by introducing self-attention, which allows models to understand word context based on surrounding words.

This led to the development of Large Language Models (LLMs).

======================
LARGE LANGUAGE MODELS (LLMs)
======================

LLMs are advanced deep learning models trained on enormous datasets of text. They learn how language is structured, how sentences flow, and how ideas connect. This allows them to generate fluent, context-aware responses.

Examples:
- GPT (OpenAI)
- LLaMA (Meta)
- Claude (Anthropic)
- Gemini (Google)

LLMs are trained using next-word prediction. By predicting the next word in billions of sentences, the model learns grammar, reasoning patterns, world knowledge, and contextual understanding.

LLMs can:
- Answer questions
- Create summaries
- Explain complex topics
- Write content
- Generate code
- Assist in research and analysis

======================
GENERATIVE AI
======================

Generative AI refers to models that create new content rather than simply analyzing or classifying existing data. They generate text, images, music, or even video.

Examples:
- GPT generating articles
- DALL·E and Stable Diffusion generating images
- MusicLM generating music tracks

Generative AI works by learning patterns and structures from training data and then generating new variations that follow similar patterns.

Uses:
- Content creation
- Personalized learning systems
- Game/film character design
- Data augmentation
- Creative tools for art and entertainment

======================
AGENTIC AI (AI AGENTS)
======================

Agentic AI models do not just respond — they take action, plan, and decide autonomously.

Agentic AI systems:
- Break tasks into smaller steps
- Reason about the best approach
- Call external tools (browsers, APIs, databases)
- Validate their own output
- Improve over time automatically

Examples:
- AI personal assistants that schedule calls
- Research agents that summarize scientific papers
- Autonomous business workflow agents

Agentic AI is important because it turns AI from a “chatbot” into a true assistant.

======================
RETRIEVAL AUGMENTED GENERATION (RAG)
======================

LLMs sometimes hallucinate — meaning they make up information. RAG solves this.

RAG does:
1) Retrieve **relevant facts** from a knowledge stored in a vector database (like Pinecone)
2) Feed those facts into the model before generating the answer

This ensures accurate, trustworthy responses grounded in your data.

======================
CONCLUSION
======================

AI → ML → Deep Learning → Transformers → LLMs → Generative + Agentic AI  
This is the evolution of intelligence in machines.

Your chatbot uses this knowledge + your Pinecone database to answer accurately and powerfully.
"""


knowledge_doc = Document(
    page_content=ai_knowledge,
    metadata={"source": "AI_Knowledge_Base"}
)


# ----------------- SPLIT LARGE TEXT INTO CHUNKS -----------------
chunked_knowledge_docs = splitter.split_documents([knowledge_doc])

all_docs_to_add = profile_docs + chunked_knowledge_docs


# ----------------- ADD TO PINECONE -----------------
print("Uploading documents to Pinecone...")
vector_store.add_documents(all_docs_to_add)

print(f"Upload Complete!")
print(f"Total documents added: {len(all_docs_to_add)}")
