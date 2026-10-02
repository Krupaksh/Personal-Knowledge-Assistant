## IMPORT REQUIRED LIBRARIES
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
from ollama import chat


## DOCUMENT CONVERSION

converter = DocumentConverter()

converted_docs = []
file_name=[]

# Specify the folder containing the PDF files.
documents_folder = Path(__file__).resolve().parent / "Data" / "Documents"

pdf_files = list(documents_folder.glob("*.pdf"))


for pdf_file in pdf_files:
    file_name.append(pdf_file.name)
    doc = converter.convert(pdf_file).document
    converted_docs.append(doc)


print("Number of converted documents:", len(converted_docs))


## CHUNKING

chunker = HybridChunker()


all_chunks = []


chunk_sources = []

for name, doc in zip(file_name, converted_docs):

    chunks = list(chunker.chunk(doc))

    all_chunks.extend(chunks)

    chunk_sources.extend([name] * len(chunks))

print("Total number of chunks:", len(all_chunks))



## EMBEDDING MODEL

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


## EXTRACT CHUNK TEXT

texts = []

for chunk in all_chunks:
    texts.append(chunk.text)


## EMBED ALL CHUNKS

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)


## CREATE VECTOR INDEX

index = faiss.IndexFlatIP(384)
index.add(embeddings)


## RETRIEVAL FUNCTION

def retrieve(question, model, index, top_k, all_chunks, chunk_sources):

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    question_embedding_2d = question_embedding.reshape(1, 384)

    distances, indices = index.search(
        question_embedding_2d,
        top_k
    )

    # BUILD RETRIEVAL RESULTS

    results = []

    for i, score in zip(indices[0], distances[0]):

        chunk = all_chunks[i]
        source = chunk_sources[i]

        pages = []

        for item in chunk.meta.doc_items:
            for provenance in item.prov:
                pages.append(provenance.page_no)

        unique_pages = sorted(set(pages))

        result = {
            "score": score,
            "pages": unique_pages,
            "text": chunk.text,
            "source": source
        }

        results.append(result)

    return results


## ASK MULTIPLE QUESTIONS

while True:

    # Ask the user to enter a question.
    question = input("\nEnter your question (or type 'exit' to quit): ")

    # Exit the program if the user types 'exit'.
    if question.lower() == "exit":
        print("Exiting Knowledge Assistant.")
        break

    # CALL THE RETRIEVAL FUNCTION
    results = retrieve(question, model, index, 3, all_chunks, chunk_sources)


    # EXTRACT RETRIEVED TEXT FOR THE LLM

    retrieved_texts = []

    for result in results:

        page_text = ", ".join(str(page) for page in result["pages"])

        retrieved_texts.append(
        f"Source: {result['source']} | Pages: {page_text}\n{result['text']}"
        )


    # COMBINE RETRIEVED CHUNKS INTO ONE CONTEXT

    context = "\n\n".join(retrieved_texts)


    ## BUILD GROUNDED LLM PROMPT

    # Instruct the LLM to answer using only the retrieved context.
    prompt = f"""
    You are a teacher who gives correct and precise answers.

    Answer the question using only the provided context.

    If the context does not contain sufficient information to answer
    the question correctly, say:
    "The context provided isn't sufficient for me to give a correct answer."

    Context:

    {context}

    Question:

    {question}
    """


    ## GENERATE ANSWER USING OLLAMA

    response = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # DISPLAY ANSWER

    print("\nAnswer:\n")
    print(response.message.content)


    # DISPLAY SOURCE PAGE NUMBERS

    # DISPLAY SOURCE DOCUMENTS AND PAGE NUMBERS

    print("\nSources:")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['source']} | Page(s): {result['pages']}"
        )