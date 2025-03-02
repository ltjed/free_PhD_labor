import json
import os
import re
import os.path as osp

import anthropic
import backoff
import openai
from transformers import GPT2Tokenizer





MAX_NUM_TOKENS = 4096
idea_str_archive = []
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
    response = client.messages.create(
        model=model,
        max_tokens=MAX_NUM_TOKENS,
        temperature=temperature,
        system=system_message,
        messages=new_msg_history,
    )
    content = response.content[0].text
    new_msg_history = new_msg_history + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    ]
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history


def create_client(model):
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif 'gpt' in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model == "deepseek-coder-v2-0724":
        print(f"Using OpenAI API with {model}.")
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        ), model
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ), "meta-llama/llama-3.1-405b-instruct"
    else:
        raise ValueError(f"Model {model} not supported.")
    
import PyPDF2
import re
from transformers import pipeline


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF using PyPDF2.
    Returns a single string of all text from the PDF.
    """
    text = ""
    title = None
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        info = reader.metadata
        if info and info.get('/Title'):
            return info['/Title']
            
        # If no title in metadata, try to extract from first page
        if len(reader.pages) > 0:
            first_page = reader.pages[0].extract_text()
            # Look for first line of text, assuming it's the title
            lines = first_page.split('\n')
            for line in lines:
                line = line.strip()
                if line:  # First non-empty line
                    title = line                    
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text, title


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
from transformers import GPT2TokenizerFast

def two_stage_chunking(
    text: str,
    word_chunk_size: int = 256,
    max_tokens: int = 1024,
    tokenizer_name: str = "gpt2",
    add_special_tokens: bool = False
):
    """
    1. Splits `text` into chunks of `word_chunk_size` words each (e.g., 512).
    2. For each word-based chunk, tokenize it and split further (if needed)
       so that each final chunk has at most `max_tokens` tokens.
    
    Returns a list of string chunks, each of which encodes to <= `max_tokens`.
    """
    # 1. Split the text by words so we don't handle a massive string at once.
    words = text.split()
    
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    
    final_chunks = []
    
    # 2. Loop over word-based chunks
    for i in range(0, len(words), word_chunk_size):
        word_chunk = words[i : i + word_chunk_size]
        sub_text = " ".join(word_chunk)
        
        # -- Tokenize this sub-text
        token_ids = tokenizer.encode(sub_text, add_special_tokens=add_special_tokens)
        
        # -- If it exceeds max_tokens, break it down further
        start = 0
        while start < len(token_ids):
            end = start + max_tokens
            # Grab a slice of tokens (bounded by max_tokens)
            token_slice = token_ids[start:end]
            # Decode back to text
            chunk_str = tokenizer.decode(token_slice, skip_special_tokens=not add_special_tokens)
            final_chunks.append(chunk_str)
            start = end
    
    return final_chunks

def chunk_text_by_tokens(text, max_tokens=512, tokenizer_name="gpt2"):
    """
    Splits 'text' into chunks of up to 'max_tokens' *model tokens* each
    using a Hugging Face tokenizer (e.g., GPT2TokenizerFast).
    
    Args:
        text (str): The input text to be tokenized and chunked.
        max_tokens (int): The maximum number of tokens allowed in each chunk.
        tokenizer_name (str): The model name of the tokenizer to use.

    Returns:
        List[str]: A list of text chunks, each at most 'max_tokens' tokens long.
    """
    # 1. Initialize a Hugging Face tokenizer (GPT2 is just an example).
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    # 2. Encode the text into a list of token IDs (disable special tokens for naive splitting).
    inputs = tokenizer.encode(text, add_special_tokens=False)
    # 3. Loop through the token list in steps of 'max_tokens'.
    chunks = []
    for i in range(0, len(inputs), max_tokens):
        token_chunk = inputs[i : i + max_tokens]
        # Decode the tokens back to a string
        chunk_text = tokenizer.decode(token_chunk, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


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

def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

def summarize_in_chunks(text, summarizer, max_tokens_per_chunk=1024, max_length=200, min_length=50):
    """
    1. Chunk the text into small pieces (by words).
    2. Use the summarization pipeline on each chunk.
    3. Combine all chunk summaries into a single string.
    """
    # Split into chunks
    chunks = chunk_text_by_tokens(text, max_tokens_per_chunk)
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
        print(f"Chunk summary: {chunk_summary}")
    
    # Combine chunk-level summaries into a final text
    combined_summary = " ".join(all_summaries)
    return combined_summary
    
def summarize_idea(pdf_path):
    client, client_model = create_client("claude-3-5-sonnet-20240620")
    raw_text, title = extract_pdf_text(pdf_path)

    # Step 2: Clean text
    cleaned_text = clean_text(raw_text)
    # Print the length of the cleaned text in number of characters
    print(f"Number of characters in cleaned text: {len(cleaned_text)}")
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
    
    print(f"Length of short summary: {len(short_summary.split())} words")
    if title == None:
        prompt = """ You are given a summary of a paper: 

        {summary}

        In <JSON>, read and evaluate the paper summary, then record your summary and evaluation in JSON format with the following fields:
        - "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
        - "Title": A title for the idea, will be used for the report writing.
        - "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
        - "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms.
        - "Research_Impact": Identify a specific challenge in latest mechanistic interpretability literature. Then, explain how the proposed research address that challenge, citing specific issues and resolutions.
        - "Interestingness": A rating from 1 to 10 (lowest to highest).
        - "Feasibility": A rating from 1 to 10 (lowest to highest).
        - "Novelty": A rating from 1 to 10 (lowest to highest).

        Be cautious and critical on your ratings.
        This JSON will be automatically parsed, so ensure the format is precise.
        """
        prompt = prompt.format(summary=short_summary)
    else:
        prompt = """ You are given a summary of a paper:

        {summary}

        In <JSON>, read and evaluate the paper summary, then record your summary and evaluation in JSON format with the following fields:
        - "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
        - "Title": A title for the idea, will be used for the report writing.
        - "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
        - "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms.
        - "Research_Impact": Identify a specific challenge in latest mechanistic interpretability literature. Then, explain how the proposed research address that challenge, citing specific issues and resolutions.
        - "Interestingness": A rating from 1 to 10 (lowest to highest).
        - "Feasibility": A rating from 1 to 10 (lowest to highest).
        - "Novelty": A rating from 1 to 10 (lowest to highest).

        Be cautious and critical on your ratings.
        This JSON will be automatically parsed, so ensure the format is precise.
        
        The title is known to be: {title} so simply fill in the Title field but fill in the rest of the JSON as you see fit. 
        """
        prompt = prompt.format(summary=short_summary, title=title)
    response, _ = get_response_from_llm(
        prompt,
        client,
        client_model,
        system_message="You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.",
        msg_history=[]
        )

    json_output = extract_json_between_markers(response)
    return json_output

import os
import glob

# Get all PDF files in the example_papers folder
# can change to other templates as well given they have a literature_context folder
pdf_files = glob.glob("templates/sae_variants/literature_context/*.pdf")

# Initialize idea_str_archive if it doesn't exist
idea_str_archive = []

# Process each PDF file
for pdf_path in pdf_files:
    try:
        # Get JSON output from summarize_idea
        json_output = summarize_idea(pdf_path)
        
        # Convert to string and append to archive
        if json_output:
            idea_str_archive.append(json.dumps(json_output))
            print(f"Successfully processed {pdf_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
ideas = []

for idea_str in idea_str_archive:
    ideas.append(json.loads(idea_str))

with open(osp.join("templates/sae_variants", "seed_ideas.json"), "w") as f:
    json.dump(ideas, f, indent=4)
