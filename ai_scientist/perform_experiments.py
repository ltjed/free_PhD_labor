import json
import os.path as osp
import os
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
from ai_scientist.llm import get_response_from_llm
MAX_ITERS = 10 # originally 10
MAX_RUNS = 10 # originally 5
MAX_STDERR_OUTPUT = 1500
NUM_EXPERIMENT_REFLECTIONS = 5
coder_prompt = """Your goal is to implement the following idea: {title}. Pay attention to the following details from the idea:
The proposed experiment is as follows: {idea}.
The implementation plan is as follows: {implementation_plan}.

You can also refer to other information in the idea: {context_information}

You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

If the experiments in the idea is already implemented in 'experiment.py' you are given with, you should try to improve its result by further enhancing the implementation.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run (you can try to run with different hyperparameters in the same run across different iterations.).

Note that we already provide the baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list."""

reflected_coder_prompt = """Your goal is to implement the following idea: {title}. Pay attention to the following details from the idea:
The proposed experiment is as follows: {idea}.
The implementation plan is as follows: {implementation_plan}.

You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run (you can try to run with different hyperparameters in the same run across different iterations.).

Note that we already provide the baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list."""


first_reflection_prompt = """\
Below is the current experiment idea, its results, baseline results, and any relevant notes.

==== EXPERIMENT IDEA ====
{idea}

==== EXPERIMENT RESULTS ====
{results}

==== BASELINE RESULTS ====
{baseline_results}

==== NOTES ====
{notes}

THOUGHT:
- Compare these experiment results to the baseline and results from previous runs in NOTES.
- Discuss whether they match or contradict expectations, and why.
- Try to explain the matching and unmatching of expectations with theoretical insights.
- If you decide changes or further exploration are needed, describe them.

The correct interpretation for scores are as follows: 
For absoprtion, a lower "mean_absorption_score" means better performance of the underlying SAE in the run. Generally, a "mean_absorption_score" < 0.01 is considered a good target score.

For unlearning, a higher score indicate better performance in unlearning dangerous knowledge and thus considered better. Generally, a "unlearning_score" > 0.1 is considered a good target score.

For sparse probing, a higher "sae_top_1_test_accuracy" score indicates better performance of the underlying SAE in the run. Generally, a "sae_top_1_test_accuracy" > 0.74 is considered a good target score

For autointerp, a higher score means better performance of the underlying SAE in the run.

"""

next_reflection_prompt = """\
Below is your previous reflection and plan. Revisit and refine it if necessary.

==== PREVIOUS REFLECTION ====
{previous_reflection}

==== BASELINE RESULTS ====
{baseline_results}

==== NOTES ====
{notes}

THOUGHT:
- Re-examine your prior reasoning and plan considering the baseline and notes.
- If you conclude further changes are required, detail them.
- If no further reflection is needed, end your THOUGHT with 'I am done'
  and provide your final plan in <FINAL_PLAN> ... </FINAL_PLAN>.

The correct interpretation for scores are as follows: 
For absoprtion, a lower "mean_absorption_score" means better performance of the underlying SAE in the run. Generally, a "mean_absorption_score" < 0.01 is considered a good target score.

For unlearning, a higher score indicate better performance in unlearning dangerous knowledge and thus considered better. Generally, a "unlearning_score" > 0.1 is considered a good target score.

For sparse probing, a higher "sae_top_1_test_accuracy" score indicates better performance of the underlying SAE in the run. Generally, a "sae_top_1_test_accuracy" > 0.74 is considered a good target score

For autointerp, a higher score means better performance of the underlying SAE in the run.

"""


system_prompt = """\
You are an independent ML research expert, providing iterative feedback on an experiment idea. Consider carefully if you want the experiments to be re-planned given the result from this run. This could mean either merely changing hyperparameters or change of implementation of the SAE architecture.

At each step:
- Present your reasoning in a 'THOUGHT:' section.
- Compare the experiment's results to any baseline results, noting whether they align with expectations.
- Suggest whether changes or further experimentation are needed.
- If you conclude no further reflection is required, end your THOUGHT with 'I am done' 
  and produce your final plan in this format:

<FINAL_PLAN>
(Here describe everything that should happen next, including any improvements or new steps.)
</FINAL_PLAN>

The final plan must capture all key points from your reflection's conclusion.
"""


# timeout was originally set to 7200
# RUN EXPERIMENT
def run_experiment(folder_name, run_num, idea, baseline_results, client, client_model, timeout=10800):
    cwd = osp.abspath(folder_name)
    # COPY CODE SO WE CAN SEE IT.
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"run_{run_num}.py"),
    )

    # LAUNCH COMMAND
    command = [
        "python",
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]

    # initialize results variable, to be returned later to match the format of the usage in perform_experiments()

    try:
        # Delete the file if it exists
        result_file = osp.join(cwd, f"run_{run_num}", "final_info.json")
        if osp.exists(result_file):
            os.remove(result_file)
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v for k, v in results.items()}
            plan = do_reflection(idea, results, baseline_results, NUM_EXPERIMENT_REFLECTIONS, client, client_model, folder_name)
            print(f"Suggested plan:\n {plan} \n")
            next_prompt = f"""Run {run_num} completed. Here are the results:
{results}

Consider carefully if you want to re-plan your experiments given the result from this run. This could mean either merely changing hyperparameters or change of implementation of the SAE architecture.
The correct interpretation for scores are as follows: 
For absoprtion, a lower "mean_absorption_score" means better performance of the underlying SAE in the run. Generally, a "mean_absorption_score" < 0.01 is considered a good target score.

For unlearning, a higher score indicate better performance in unlearning dangerous knowledge and thus considered better. Generally, a "unlearning_score" > 0.1 is considered a good target score.

For sparse probing, a higher "sae_top_1_test_accuracy" score indicates better performance of the underlying SAE in the run. Generally, a "sae_top_1_test_accuracy" > 0.74 is considered a good target score

For autointerp, a higher score means better performance of the underlying SAE in the run.

An expert has written some comment and IMPORTANT suggestions about a plan of what to do next which you should consider and refer to: {plan}

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {run_num}, including 1. an experiment description, 2. noteworthy experiment results and 3. the run number. Be as verbose as necessary.

Then, implement the next thing on your list.
We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with experiments, respond with 'ALL_COMPLETED'."""
        return result.returncode, next_prompt

    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt


# RUN PLOTTING
def run_plotting(folder_name, timeout=600):
    cwd = osp.abspath(folder_name)
    # LAUNCH COMMAND
    command = [
        "python",
        "plot.py",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Plotting failed with return code {result.returncode}")
            next_prompt = f"Plotting failed with the following error {result.stderr}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Plotting timed out after {timeout} seconds")
        next_prompt = f"Plotting timed out after {timeout} seconds"
        return 1, next_prompt

import re

def do_reflection(idea, results, baseline_results, num_reflections, client, client_model, folder_name):
    # 1) Load notes
    with open(osp.join(folder_name, "notes.txt"), "r") as file:
        notes = file.read()

    # 3) Format the first reflection prompt
    reflection_prompt = first_reflection_prompt.format(
        idea=idea,
        results=results,
        baseline_results=baseline_results,
        notes=notes
    )

    msg_history = []
    final_plan = None

    try:
        # -- FIRST REFLECTION --
        print("Iteration 1")
        text, msg_history = get_response_from_llm(
            msg=reflection_prompt,
            system_message=system_prompt,
            client=client,
            model=client_model,
            msg_history=msg_history,
        )
        print(text)
        # Attempt to extract final plan if present
        plan_match = re.search(r"<FINAL_PLAN>([\s\S]*?)</FINAL_PLAN>", text)
        if plan_match:
            # We found a final plan in the text
            final_plan = plan_match.group(1).strip()
            return final_plan  # End function here

        previous_reflection = text

        # -- NEXT REFLECTIONS --
        for i in range(2, num_reflections + 1):
            print(f"Iteration {i}")

            reflection_prompt = next_reflection_prompt.format(
                previous_reflection=previous_reflection,
                baseline_results=baseline_results,
                notes=notes
            )

            text, msg_history = get_response_from_llm(
                msg=reflection_prompt,
                system_message=system_prompt,
                client=client,
                model=client_model,
                msg_history=msg_history,
            )
            print(text)

            # Check if done
            plan_match = re.search(r"<FINAL_PLAN>([\s\S]*?)</FINAL_PLAN>", text)
            if plan_match:
                final_plan = plan_match.group(1).strip()
                print(f"Reflection converged at iteration {i}.")
                break

            previous_reflection = text

    except Exception as e:
        print(f"Failed to reflect: {e}")

    return final_plan





# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results, client, client_model) -> bool:
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        implementation_plan = idea["Implementation_Plan"],
        context_information = idea,
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    print(f"Starting experiment with prompt for coder: {next_prompt}")
    while run < MAX_RUNS + 1:
        print(f"Currently on iteration {current_iter} of run {run}")
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        
        coder_out = coder.run(next_prompt)
        print(f"coder_out: {coder_out}, type: {type(coder_out)}")

        if "ALL_COMPLETED" in coder_out:
            break
        
        
        return_code, next_prompt = run_experiment(folder_name, run, idea, baseline_results, client, client_model)
        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Not all experiments completed.")
        return False

    current_iter = 0
    next_prompt = """
Great job! Please modify `plot.py` to generate relevant plots for the final writeup. These plots should be relevant and insightful. You should take data from the results obtained earlier in the experiments.

In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.

Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.

We will be running the command `python plot.py` to generate the plots.
"""
    while True:
        _ = coder.run(next_prompt)
        return_code, next_prompt = run_plotting(folder_name)
        current_iter += 1
        if return_code == 0 or current_iter >= MAX_ITERS:
            break
    next_prompt = """
Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth. For example, explain the significance of these figures, what is as expected, what is remarkably unexpected, etc.

Somebody else will be using `notes.txt` to write a report on this in the future.
"""
    coder.run(next_prompt)

    return True
