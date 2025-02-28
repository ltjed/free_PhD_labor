import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

from datetime import datetime

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

S2_API_KEY = os.getenv("S2_API_KEY")

benchmark_name = "unlearning"

# Maximum number of retries for JSON extraction
MAX_JSON_EXTRACTION_RETRIES = 3

# Helper function to extract JSON with retries
def extract_json_with_retries(text, client, model, system_message, msg_history, prompt, max_retries=MAX_JSON_EXTRACTION_RETRIES):
    """
    Attempts to extract JSON from text. If extraction fails, retries by asking the LLM again with the same prompt.
    
    Args:
        text: The text containing JSON to extract
        client: LLM client
        model: LLM model
        system_message: System message for the LLM
        msg_history: Message history for the LLM
        prompt: Original prompt to retry with
        max_retries: Maximum number of retries
        
    Returns:
        Extracted JSON or None if all retries fail
    """
    json_output = extract_json_between_markers(text)
    
    retry_count = 0
    while json_output is None and retry_count < max_retries:
        retry_count += 1
        print(f"Failed to extract JSON. Retry attempt {retry_count}/{max_retries}")
        
        # Simply retry with the same prompt
        text, msg_history = get_response_from_llm(
            prompt,
            client=client,
            model=model,
            system_message=system_message,
            msg_history=[]  # Reset message history to avoid confusion
        )
        
        # Try to extract JSON again
        json_output = extract_json_between_markers(text)
    
    if json_output is None:
        print("All JSON extraction retries failed")
    
    return json_output, text, msg_history

# delete the autoencoder sentence for templates other than autoencoder!
idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Now, come up with the next impactful and creative idea for improving sparse autoencoder on {benchmark_name} benchmark.
Your new idea should not be more complex than those you have already generated. DO NOT INTRODUCE ANY UNDUELY MORE COMPLEX ARCHITECTURE, UNNECESSARILY COMPLEX THEORY (ESPECIALLY MATHEMATICAL) THEORY, FUNCTIONALITY, STATISTICAL METHOD, METRIC.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first thoroughly discuss your intuitions and motivations for why your idea can improve on existing SAE on the {benchmark_name} benchmark. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. 
Provide extremely detailed reasoning on why the proposed idea will improve the target benchmark. Be as specific as possible. including technical details if necessary.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms. BE SURE TO DEFINE BEFORE YOU USE NONSTANDARD TERMINOLOGY. WHENEVER POSSIBLE, USE MATHEMATICAL LANGUAGE TO AVOID AMBIGUITY.
- "Rationale": An extremely detailed explanation of why the proposed experiment can be expected to improve from the baseline model. Carefully explaining the logic behind every step of reasoning you make. Avoid making unjustified claims about improvement.
- "Implementation_Plan": A plan of steps to implement the experiment described above by modifying the code template in experiment.py.
Be cautious and critical in your output.

This JSON will be automatically parsed, so ensure the format is precise. BE SURE TO USE ESCAPING FOR ALL SPECIAL CHARACTERS SUCH AS QUOTES, BACKSLASHES, ETC. IN THE JSON.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""


idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created, in relation to {benchmark_name} eval. Critically identify on the "Rationale" section of the idea json: is there any ambiguity or flaw in the logic or reasoning? Is there any unjustified claims about expected improvements? Think about how to resolve them step-by-step.
Ensure the idea is clear and well-justified, and the JSON is the correct format. BE SURE TO USE ESCAPING FOR ALL SPECIAL CHARACTERS SUCH AS QUOTES, BACKSLASHES, ETC. IN THE JSON.
In the next attempt, try and refine and improve your last idea.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


# GENERATE IDEAS
def generate_ideas(
        base_dir,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
                print("ideas")
            print("Loaded existing ideas:")
            # start from the last idea
            #ideas = ideas[-1:-2:-1]
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            first_prompt = idea_first_prompt.format(
                task_description=prompt["task_description"],
                code=code,
                prev_ideas_string=prev_ideas_string,
                num_reflections=num_reflections,
                benchmark_name=benchmark_name,
            )
            text, msg_history = get_response_from_llm(
                first_prompt,
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            
            ## PARSE OUTPUT with retries
            json_output, text, msg_history = extract_json_with_retries(
                text, 
                client, 
                model, 
                idea_system_prompt, 
                msg_history, 
                first_prompt
            )
            
            if json_output is None:
                raise Exception("Failed to extract JSON from LLM output after retries")
                
            print(json_output)

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    reflection_prompt = idea_reflection_prompt.format(
                        current_round=j + 2,
                        num_reflections=num_reflections,
                        benchmark_name=benchmark_name,
                    )
                    text, msg_history = get_response_from_llm(
                        reflection_prompt,
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    
                    ## PARSE OUTPUT with retries
                    json_output, text, msg_history = extract_json_with_retries(
                        text, 
                        client, 
                        model, 
                        idea_system_prompt, 
                        msg_history, 
                        reflection_prompt
                    )
                    
                    if json_output is None:
                        raise Exception("Failed to extract JSON from LLM output after retries")
                        
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
        base_dir,
        client,
        model,
        prev_idea_archive=[],
        num_reflections=5,
        max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        # changed to include all seed_ideas instead of just the first one
        for seed_idea in seed_ideas:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                first_prompt = idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ) + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
"""
                text, msg_history = get_response_from_llm(
                    first_prompt,
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                
                ## PARSE OUTPUT with retries
                json_output, text, msg_history = extract_json_with_retries(
                    text, 
                    client, 
                    model, 
                    idea_system_prompt, 
                    msg_history, 
                    first_prompt
                )
                
                if json_output is None:
                    raise Exception("Failed to extract JSON from LLM output after retries")
                    
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        reflection_prompt = idea_reflection_prompt.format(
                            current_round=j + 2, 
                            num_reflections=num_reflections
                        )
                        text, msg_history = get_response_from_llm(
                            reflection_prompt,
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        
                        ## PARSE OUTPUT with retries
                        json_output, text, msg_history = extract_json_with_retries(
                            text, 
                            client, 
                            model, 
                            idea_system_prompt, 
                            msg_history, 
                            reflection_prompt
                        )
                        
                        if json_output is None:
                            raise Exception("Failed to extract JSON from LLM output after retries")
                            
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with the following fields:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.
- "Decision": A decision on the novelty of the idea. Either "decision made: novel", "decision made: not novel", or "undecided".

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.


'''


def check_idea_novelty(
        ideas,
        base_dir,
        client,
        model,
        max_num_iterations=10,
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    # retrieve the number of seed ideas
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
    num_seed_ideas = len(seed_ideas)

    for idx, idea in enumerate(ideas):
        # Skip seed ideas
        if idx < num_seed_ideas:
            print(f"Skipping seed idea {idx}")
            idea["novel"] = False
            continue
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                novelty_prompt_formatted = novelty_prompt.format(
                    current_round=j + 1,
                    num_rounds=max_num_iterations,
                    idea=idea,
                    last_query_results=papers_str,
                )
                system_msg = novelty_system_msg.format(
                    num_rounds=max_num_iterations,
                    task_description=task_description,
                    code=code,
                )
                
                text, msg_history = get_response_from_llm(
                    novelty_prompt_formatted,
                    client=client,
                    model=model,
                    system_message=system_msg,
                    msg_history=msg_history,
                )
                
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT with retries
                json_output, text, msg_history = extract_json_with_retries(
                    text, 
                    client, 
                    model, 
                    system_msg, 
                    msg_history, 
                    novelty_prompt_formatted
                )
                
                if json_output is None:
                    raise Exception("Failed to extract JSON from LLM output after retries")

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 10
    NUM_REFLECTIONS = 5
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    args = parser.parse_args()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if args.check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )
