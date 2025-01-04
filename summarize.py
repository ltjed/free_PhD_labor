import PyPDF2
import re
from transformers import pipeline


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF using PyPDF2.
    Returns a single string of all text from the PDF.
    """
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    """
    Clean the extracted text by removing non-ASCII characters and
    other potential trouble spots that can break tokenization.
    """
    # Example: remove non-printable or extended characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # You can add more cleaning rules (e.g. extra whitespace)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, max_tokens=512):
    """
    Naive text chunking by words.
    Splits 'text' into chunks of up to 'max_tokens' words each.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            # Join the current chunk into a string
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    # Add the remainder if there's anything left
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def summarize_in_chunks(text, summarizer, max_tokens_per_chunk=512, max_length=200, min_length=50):
    """
    1. Chunk the text into small pieces (by words).
    2. Use the summarization pipeline on each chunk.
    3. Combine all chunk summaries into a single string.
    """
    # Split into chunks
    chunks = chunk_text(text, max_tokens=max_tokens_per_chunk)
    all_summaries = []
    
    for chunk in chunks:
        # Summarize each chunk
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        # Extract the actual summary string
        chunk_summary = summary[0]["summary_text"]
        all_summaries.append(chunk_summary)
    
    # Combine chunk-level summaries into a final text
    combined_summary = " ".join(all_summaries)
    return combined_summary


def generate_paper_summary(pdf_path):
    """
    Main function:
    1. Extract & clean PDF text
    2. Summarize the text in chunks (on GPU device=0)
    3. Return a structured dict with final results
    """
    # Step 1: Extract text
    raw_text = extract_pdf_text(pdf_path)
    
    # Step 2: Clean text
    cleaned_text = clean_text(raw_text)
    
    # Step 3: Create a summarization pipeline on GPU
    #    device=0 => GPU; adjust if you have multiple GPUs (e.g., device=1 for second GPU).
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0
    )
    
    # Step 4: Summarize in chunks
    short_summary = summarize_in_chunks(
        cleaned_text,
        summarizer,
        max_tokens_per_chunk=512,
        max_length=200,
        min_length=50
    )
    
    # Step 5: Construct a JSON-like structured output
    # In a real system, you could parse "short_summary" or use more advanced methods to fill these fields automatically.
    # For demonstration, let's just place the summary into 'Experiment' or use it however you see fit.
    structured_summary = {
        "Name": "sparse_autoencoder_scaling",
        "Title": "Scaling Laws and Evaluation Methods for Sparse Autoencoders in Language Models",
        "Experiment": short_summary,  # Put the final summary here (or adapt as needed)
        "Interestingness": 8,
        "Feasibility": 7,
        "Novelty": 7
    }
    
    return structured_summary


if __name__ == "__main__":
    # Replace with the path to your PDF
    pdf_file_path = "example_paper.pdf"
    
    summary_dict = generate_paper_summary(pdf_file_path)
    print(summary_dict)
