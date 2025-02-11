import argparse
import json
import os.path as osp
import openai
from datetime import datetime
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate writeup from existing results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the results directory containing experiment results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        help="Model to use for writeup generation",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews",
    )
    return parser.parse_args()

def generate_writeup_from_results(results_dir, model, improvement=False):
    folder_name = results_dir
    idea_name = osp.basename(folder_name)
    
    # Load idea details from notes.txt
    notes_path = osp.join(folder_name, "notes.txt")
    with open(notes_path, 'r') as f:
        notes_content = f.read()
    
    # Parse title and experiment description
    title = None
    experiment = None
    for line in notes_content.split('\n'):
        if line.startswith('# Title:'):
            title = line.replace('# Title:', '').strip()
        elif line.startswith('# Experiment description:'):
            experiment = line.replace('# Experiment description:', '').strip()
    
    idea = {
        'Name': idea_name,
        'Title': title,
        'Experiment': experiment
    }

    # Setup for writeup generation
    exp_file = osp.join(folder_name, "experiment.py")
    writeup_file = osp.join(folder_name, "latex", "template.tex")
    notes = osp.join(folder_name, "notes.txt")
    fnames = [exp_file, writeup_file, notes]
    
    io = InputOutput(
        yes=True,
        chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
    )
    
    main_model = Model(model)
    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )

    # Create client for the model
    if model.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
    else:
        client = openai.OpenAI()

    try:
        print(f"Starting writeup generation for {idea_name}")
        perform_writeup(idea, folder_name, coder, client, model)
        
        print("Generating PDF...")
        generate_latex(coder, folder_name, f"{folder_name}/{idea['Name']}.pdf")
        
        print("Performing review...")
        paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
        review = perform_review(
            paper_text,
            model="gpt-4o-2024-05-13",
            client=openai.OpenAI(),
            num_reflections=5,
            num_fs_examples=1,
            num_reviews_ensemble=5,
            temperature=0.1,
        )
        
        with open(osp.join(folder_name, "review.txt"), "w") as f:
            f.write(json.dumps(review, indent=4))
            
        if improvement:
            print("Starting improvement based on review...")
            perform_improvement(review, coder)
            generate_latex(
                coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
            )
            
            paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
            review = perform_review(
                paper_text,
                model="gpt-4o-2024-05-13",
                client=openai.OpenAI(),
                num_reflections=5,
                num_fs_examples=1,
                num_reviews_ensemble=5,
                temperature=0.1,
            )
            
            with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                f.write(json.dumps(review, indent=4))
                
        print("Writeup generation completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during writeup generation: {e}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    success = generate_writeup_from_results(args.results_dir, args.model, args.improvement)
    if success:
        print("Writeup generation completed successfully")
    else:
        print("Writeup generation failed")