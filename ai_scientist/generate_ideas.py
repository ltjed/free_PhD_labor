import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

S2_API_KEY = os.getenv("S2_API_KEY")

# delete the autoencoder sentence for templates other than autoencoder!
idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Come up with an impactful and creative improvement idea from the following previous result.

Here is the previous idea from which you should develop and improve. The idea you come up with should NOT be a completely different idea from the previous idea, but should be more mature development. DO NOT INTRODUCE ANY UNDUELY MORE COMPLEX ARCHITECTURE, UNNECESSARILY COMPLEX THEORY (ESPECIALLY MATHEMATICAL) THEORY, FUNCTIONALITY, STATISTICAL METHOD, TECHNIQUE, METIC, OR NONSTANDARD TRAINING SCHEMES THAT ARE NOT CONTAINED (explicit or implicit) IN THE PROTOTYPE IDEA. THAT IS, GO DEEPER, NOT WIDER.

<PROTOTYPE_IDEA>
        "Name": "sparse_orthogonal_sae",
        "Title": "Sparsity-Guided Orthogonality Constraints for Interpretable Feature Separation",
        "Experiment": "1. Use existing sparsity masks to identify competing features\n2. Add sparsity-weighted orthogonality loss\n3. Train on google/gemma-2-2b using standard datasets\n4. Compare benchmark performance against baseline and other orthogonal SAEs\n5. Analyze feature competition patterns\n6. Evaluate impact of competition thresholds",
        "Technical_Details": "The method uses a sparsity-based orthogonality loss: L = L_recon + \u03bb_1 * L_sparse + \u03bb_2 * \u03a3_(i,j) c_(ij) * |f_i^T f_j| where c_(ij) is the normalized intersection size of sparsity masks for features i and j over a batch. Features that frequently activate on the same inputs face stronger orthogonality constraints, encouraging them to learn distinct concepts. The competition coefficients c_(ij) are computed directly from the existing top-k activation masks with no additional overhead.",
        "Implementation_Plan": "1. Add function to compute mask intersections from top-k indices\n2. Modify AutoEncoderTopK to use sparsity patterns\n3. Add sparsity-weighted orthogonality loss\n4. Add configuration for competition threshold\n5. Add evaluation metrics for feature competition\n6. Update training loop to use activation masks",
        "Interestingness_Evaluation": "Using sparsity patterns to guide orthogonality provides a direct and elegant connection between the two key mechanisms for feature separation.",
        "Interestingness": 9,
        "Feasibility_Evaluation": "Implementation uses only existing top-k masks; no additional computation needed; standard matrix operations; easily within 30-minute limit on H100; minimal code changes required.",
        "Feasibility": 10,
        "Novelty_Evaluation": "Leveraging sparsity patterns to guide orthogonality constraints is a novel and principled approach that directly targets feature competition.",
        "Novelty": 9,
        "Expected_Research_Impact": "The direct connection between sparsity and orthogonality should provide more interpretable feature separation while maintaining computational efficiency.",
        "Research_Impact": 9,
        "Overall_Score": 9.4,
</PROTOTYPE_IDEA>


Here is information from the previous experiment log:

<log>
# Title: Sparsity-Guided Orthogonality Constraints for Interpretable Feature Separation
# Experiment description: 1. Use existing sparsity masks to identify competing features
2. Add sparsity-weighted orthogonality loss
3. Train on google/gemma-2-2b using standard datasets
4. Compare benchmark performance against baseline and other orthogonal SAEs
5. Analyze feature competition patterns
6. Evaluate impact of competition thresholds

# Generated Figures Analysis

## absorption_comparison.png
This figure shows the mean absorption scores across different model configurations. Absorption scores measure how well individual features capture specific concepts. Key observations:
- Run 4 (optimal dictionary size) achieved the highest absorption score (0.025), nearly 3x the baseline (0.009)
- Increasing orthogonality weight alone (Runs 1-3) showed steady improvement in absorption
- Dictionary size increase beyond optimal (Run 5) led to decreased absorption, suggesting feature dilution
- Strong orthogonality with optimal dictionary size (Run 6) maintained good absorption but didn't exceed Run 4

## scr_comparison.png 
Shows Sparsity-Constrained Reconstruction (SCR) metrics at k=2 and k=20 thresholds. SCR measures feature selectivity and independence. Notable findings:
- All orthogonal variants showed improved SCR scores over baseline
- Run 4's configuration achieved best SCR metrics (0.172 at k=2), indicating cleaner feature separation
- Higher orthogonality weights correlated with better SCR scores up to a point
- Larger dictionary sizes didn't necessarily improve feature selectivity
- The gap between k=2 and k=20 metrics narrowed with orthogonality, suggesting more consistent feature behavior

## reconstruction_quality.png
Compares MSE and cosine similarity metrics for reconstruction quality. Unexpected findings:
- Despite stronger constraints, reconstruction quality remained remarkably stable across all runs
- MSE stayed consistently around 1.41 with minimal variation
- Cosine similarity maintained ~0.93 even with highest orthogonality weight
- No significant degradation with increased dictionary size
- The stability suggests orthogonality constraints don't compromise reconstruction ability

## sparse_probing.png
Shows top-1 and top-20 accuracy for sparse probing tasks. Key insights:
- All orthogonal variants improved over baseline probing accuracy
- Run 4 achieved best balance of top-1 (0.961) and top-20 (0.959) accuracy
- Larger dictionary sizes maintained high accuracy but didn't provide additional benefits
- Strong orthogonality (Run 6) preserved probing performance while improving interpretability
- The small gap between top-1 and top-20 accuracy suggests high feature precision

Overall, the figures demonstrate that sparsity-guided orthogonality constraints with optimal dictionary size (Run 4) achieve the best balance of:
- Improved feature separation (absorption and SCR metrics)
- Maintained reconstruction quality
- Enhanced interpretability (probing accuracy)
- Efficient resource usage (dictionary size)

The results suggest that careful tuning of orthogonality constraints and dictionary size can significantly improve feature disentanglement without sacrificing model performance.

## Run 1: Initial Orthogonality Test
Description: Testing orthogonality loss with weight=0.01 to establish baseline behavior
Results:
- Core metrics show good reconstruction (mse=1.41, cossim=0.93) with expected sparsity (L0=320)
- Absorption scores improved vs baseline (0.019 vs 0.009) suggesting better feature separation
- SCR metrics show stronger feature selectivity (scr_dir1_threshold_2=0.196 vs 0.132 baseline)
- Sparse probing accuracy improved (0.961 vs 0.958) indicating maintained interpretability
- Orthogonality loss successfully reduced feature competition while preserving performance

## Run 2: Increased Orthogonality Weight
Description: Testing stronger orthogonality constraint with weight=0.1 to analyze tradeoffs
Results:
- Reconstruction quality remained good but slightly decreased (mse=1.41, cossim=0.93) compared to Run 1
- SCR metrics showed substantial improvement (scr_dir1_threshold_2=0.158 vs 0.132 baseline)
- Sparse probing accuracy improved further (0.960 vs 0.958 Run 1)
- Absorption scores maintained (0.011 vs 0.009 baseline) despite stronger orthogonality
- Feature competition reduced while maintaining interpretability
- Higher orthogonality weight successfully increased feature separation without major performance tradeoffs

## Run 3: Balanced Orthogonality
Description: Testing moderate orthogonality constraint with weight=0.05 to find optimal balance
Results:
- Core metrics show good reconstruction (mse=1.41, cossim=0.93) with expected sparsity (L0=320)
- Absorption scores significantly improved (0.0215 vs 0.009 baseline) indicating better feature separation
- SCR metrics show substantial improvement (scr_dir1_threshold_2=0.181 vs 0.132 baseline)
- Sparse probing accuracy improved (0.959 vs 0.951 baseline) demonstrating maintained interpretability
- Balanced orthogonality weight successfully improved feature separation while preserving performance
- Feature competition reduced more effectively than Run 1 while avoiding Run 2's reconstruction tradeoffs

## Run 4: Increased Dictionary Size
Description: Testing orthogonality loss with weight=0.075 and increased dictionary size (18432 vs 2304) to analyze capacity-competition tradeoff
Results:
- Core metrics maintained good reconstruction (mse=1.41, cossim=0.93) with target sparsity (L0=320)
- Absorption scores showed strongest improvement yet (0.025 vs 0.009 baseline) indicating enhanced feature separation
- SCR metrics reached best performance (scr_dir1_threshold_2=0.172 vs 0.132 baseline)
- Sparse probing accuracy significantly improved (0.961 vs 0.951 baseline) showing better interpretability
- Increased dictionary size with stronger orthogonality successfully reduced feature competition
- Higher capacity allowed features to specialize more effectively while maintaining reconstruction quality
- Results suggest larger dictionaries can help balance sparsity and orthogonality constraints

## Run 5: Further Dictionary Size Increase
Description: Testing orthogonality loss with weight=0.075 and further increased dictionary size (32768 vs 18432) to explore capacity scaling limits
Results:
- Core metrics remained stable (mse=1.40, cossim=0.93) maintaining target sparsity (L0=320)
- Absorption scores decreased slightly (0.017 vs 0.025 previous) but still above baseline
- SCR metrics showed slight decline (scr_dir1_threshold_2=0.125 vs 0.172 previous)
- Sparse probing accuracy maintained improvement (0.959 vs 0.951 baseline)
- Larger dictionary size did not yield additional benefits for feature separation
- Results suggest optimal dictionary size around 18432 features for this configuration
- Further increases may not improve performance without adjusting other hyperparameters

## Run 6: Optimal Dictionary Size with Strong Orthogonality
Description: Testing orthogonality loss with increased weight=0.1 and optimal dictionary size (18432) to maximize feature separation
Results:
- Core metrics showed slight degradation (mse=1.41, cossim=0.93) while maintaining target sparsity (L0=320)
- Absorption scores improved (0.012 vs 0.009 baseline) but lower than Run 4's peak
- SCR metrics showed strong improvement (scr_dir1_threshold_2=0.158 vs 0.132 baseline)
- Sparse probing accuracy significantly improved (0.961 vs 0.951 baseline)
- Higher orthogonality weight successfully increased feature competition reduction
- Results suggest trade-off between reconstruction quality and feature separation
- Optimal configuration appears to be Run 4's parameters (dict_size=18432, ortho_weight=0.075)
</log>

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. **Justify how the idea refines the prototype idea without introducing too much complexity.**
Also detail the reasoning behind why they expect the modification of autoencoder you propose will work better for mechanistic interpretability purposes.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms.
- "Implementation_Plan": A plan of steps to implement the experiment described above by modifying the code template in experiment.py.
- "Interestingness_Evaluation": Give a one-sentence evaluation of the interestingness of the idea to justify the rating below.  
- "Interestingness": A rating from 1 to 10 (lowest to highest). 
- "Feasibility_Evaluation": Carefully evaluate the feasibility to justify the rating below. BE STRICT. Consider the following three factors. 1. Refer to the "Experiment" and "Technical_Details" fields you wrote above, and consider the complexity of the idea in comparison to typical ML conference papers. Note it need not and should better not introduce too deep theoretical elements such as those from pure mathematics. Be relevant and understandable to the machine learning community. 2. Refer to the "Implementation_Plan" you wrote and consider the implementation difficulties. Note the coding work to implement the experiments is to be completed by a junior CS PhD student within 1 month. 3. Refer to the "Experiment", "Technical_Details", and "Implementation_Plan" and consider the time cost of running the experiment. Note each run of the experiment has to be conducted on a single NVIDIA H100 GPU WITHIN 30 MINS.
- "Feasibility": A rating from 1 to 10 (lowest to highest). 
- "Novelty_Evaluation": Give a one-sentence evaluation of the novelty of the idea to justify the rating below. 
- "Novelty": A rating from 1 to 10 (lowest to highest).
- "Expected_Research_Impact": Your primary target is to improve performance on the benchmarks "sparse_probing" and "core". Evaluate your expectation of whether the proposed model and experiment are promising to perform well on this benchmark.
- "Research_Impact": A rating from 1 to 10 (lowest to highest).
- "Overall_Score": A single number rating computed by 0.1 * Interestingness + 0.4 * Feasibility + 0.2 * Novelty + 0.3 * Rsearch_Impact. DO NOT INCLUDE THE COMPUTATION.
- "Abstract": An abstract of the idea, which will be used for the report writing. The style, length, and content should be similar to a conference paper abstract. **BUT OMIT ALL RESULTS ABOUT IMPROVED PERFORMANCE SINCE THE IDEA HAS NOT BEEN IMPLEMENTED YET (EVEN IF YOU EXPECT SUCH RESULTS).**

Be cautious and critical on your ratings.

This JSON will be automatically parsed, so ensure the format is precise. BE SURE TO USE ESCAPING FOR ALL SPECIAL CHARACTERS SUCH AS QUOTES, BACKSLASHES, ETC. IN THE JSON.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""


xxx = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research and experiments.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.
Also detail the reasoning behind why they expect the modification of autoencoder you propose will work better for mechanistic interpretability purposes.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms.
- "Implementation_Plan": A plan of steps to implement the experiment described above by modifying the code template in experiment.py.
- "Interestingness_Evaluation": Give a one-sentence evaluation of the interestingness of the idea to justify the rating below.  
- "Interestingness": A rating from 1 to 10 (lowest to highest). 
- "Feasibility_Evaluation": Carefully evaluate the feasibility to justify the rating below. BE STRICT. Consider the following three factors. 1. Refer to the "Experiment" and "Technical_Details" fields you wrote above, and consider the complexity of the idea in comparison to typical ML conference papers. Note it need not and should better not introduce too deep theoretical elements such as those from pure mathematics. Be relevant and understandable to the machine learning community. 2. Refer to the "Implementation_Plan" you wrote and consider the implementation difficulties. Note the coding work to implement the experiments is to be completed by a junior CS PhD student within 1 month. 3. Refer to the "Experiment", "Technical_Details", and "Implementation_Plan" and consider the time cost of running the experiment. Note each run of the experiment has to be conducted on a single NVIDIA H100 GPU WITHIN 30 MINS.
- "Feasibility": A rating from 1 to 10 (lowest to highest). 
- "Novelty_Evaluation": Give a one-sentence evaluation of the novelty of the idea to justify the rating below. 
- "Novelty": A rating from 1 to 10 (lowest to highest).
- "Expected_Research_Impact": Your primary target is to improve performance on the benchmarks "sparse_probing" and "core". Evaluate your expectation of whether the proposed model and experiment are promising to perform well on this benchmark.
- "Research_Impact": A rating from 1 to 10 (lowest to highest).
- "Overall_Score": A single number rating computed by 0.2 * Interestingness + 0.4 * Feasibility + 0.2 * Novelty + 0.2 * Rsearch_Impact. DO NOT INCLUDE THE COMPUTATION. Note a 9.0 score would yield an oral presentation at a top-tier conference, while a 7.5 score would yield a poster presentation at a top-tier conference.
- "Abstract": An abstract of the idea, which will be used for the report writing. The style, length, and content should be similar to a conference paper abstract. **BUT OMIT ALL RESULTS ABOUT IMPROVED PERFORMANCE SINCE THE IDEA HAS NOT BEEN IMPLEMENTED YET (EVEN IF YOU EXPECT SUCH RESULTS).**

Be cautious and critical on your ratings.

This JSON will be automatically parsed, so ensure the format is precise. BE SURE TO USE ESCAPING FOR ALL SPECIAL CHARACTERS SUCH AS QUOTES, BACKSLASHES, ETC. IN THE JSON.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""




idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created, especially the "Overall_Score" which should be at least 8.5, and each other rating should be at least 8.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format. BE SURE TO USE ESCAPING FOR ALL SPECIAL CHARACTERS SUCH AS QUOTES, BACKSLASHES, ETC. IN THE JSON.
In the next attempt, try and refine and improve your last idea. Stick to the spirit of the idea of TEMPORAL SAE and make sure your do not deviate too much from the prototype idea. DO NOT INTRODUCE ANY EXTRA ARCHITECTURE, UNNECESSARILY COMPLEX THEORY (ESPECIALLY MATHEMATICAL), FUNCTIONALITY, STATISTICAL METHOD, TECHNIQUE, METIC, OR NONSTANDARD TRAINING SCHEMES THAT ARE NOT CONTAINED (explicit or implicit) IN THE PROTOTYPE IDEA. THAT IS, GO DEEPER, NOT WIDER.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
**IN THE "Abstract" FIELD, BE SURE TO OMIT ALL RESULTS ABOUT IMPROVED PERFORMANCE SINCE THE IDEA HAS NOT BEEN IMPLEMENTED YET (EVEN IF YOU EXPECT SUCH RESULTS).**
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
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "Failed to extract JSON from LLM output"
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
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "Failed to extract JSON from LLM output"
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

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

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
