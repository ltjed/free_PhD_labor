import json

# generate prompt.json in the same template folder
# TODO: use the example section of the output json file in ai_scientist\generate_ideas.py

def generate_prompt_json(include_unlearning=True, include_example=True):
    """
    Generates the prompt.json file with optional features, optimized for LLM prompting.

    Args:
        include_unlearning (bool): Whether to include "Unlearning" as the target benchmark.
        include_example (bool): Whether to include the example benchmark and solution.

    Returns:
        str: The content of the prompt.json file.
    """

    base_prompt_content = {
        "system": "You are an expert AI Researcher deeply immersed in the field of mechanistic interpretability, specifically focusing on improving the interpretability of sparse autoencoders (SAEs). Your goal is to publish a paper to advance progress on a specific benchmark. Reason step-by-step how your proposed variants of SAE could improve on the target benchmark",
        "task_description": "Your current research focuses on addressing the challenge of limited interpretability in the latent space of sparse autoencoders. The core problem, as highlighted in the paper 'Applying sparse autoencoders to unlearn knowledge in language models', is polysemanticity, where individual latent features appear to represent multiple, semantically distinct concepts. This hinders our ability to understand what neural networks are truly learning.\n\nYour task is to propose novel and feasible variants of sparse autoencoders that result in more interpretable latents compared to the internal activations they are trained to reconstruct. Consider the insights from the following abstract of a key paper in this area:\n\n\"One of the roadblocks to a better understanding of neural networks' internals is polysemanticity, where neurons appear to activate in multiple, semantically distinct contexts. Polysemanticity prevents us from identifying concise, human-understandable explanations for what neural networks are doing internally. One hypothesised cause of polysemanticity is superposition, where neural networks represent more features than they have neurons by assigning features to an overcomplete set of directions in activation space, rather than to individual neurons. Here, we attempt to identify those directions, using sparse autoencoders to reconstruct the internal activations of a language model. These autoencoders learn sets of sparsely activating features that are more interpretable and monosemantic than directions identified by alternative approaches, where interpretability is measured by automated methods. Moreover, we show that with our learned set of features, we can pinpoint the features that are causally responsible for counterfactual behaviour on the indirect object identification task wang2022interpretability to a finer degree than previous decompositions. This work indicates that it is possible to resolve superposition in language models using a scalable, unsupervised method. Our method may serve as a foundation for future mechanistic interpretability work, which we hope will enable greater model transparency and steerability.\"\n\nSpecifically, you should focus on the 'Unlearning' benchmark as a key evaluation metric for your proposed variants. This benchmark, described in detail below, assesses the ability of SAEs to selectively remove knowledge while preserving general capabilities. Your proposed SAE variants should aim to achieve better performance on this benchmark compared to standard SAEs.\n\n**Unlearning Benchmark Details:**\nWe evaluate SAEs on their ability to selectively remove knowledge while maintaining model performance on unrelated tasks, following the methodology in Applying sparse autoencoders to unlearn knowledge in language models.\n\nThis SAE unlearning evaluation uses the WMDP-bio dataset, which contains multiple-choice questions containing dangerous biology knowledge. The intervention methodology involves clamping selected SAE feature activations to negative values whenever the features activate during inference. Feature selection utilizes a dual-dataset approach: calculating feature sparsity across a \"forget\" dataset (WMDP-bio corpus) and a \"retain\" dataset (WikiText). The selection and intervention process involves three key hyperparameters:\n\n* `retain_threshold`: maximum allowable sparsity on the retain set\n* `n_features`: number of top features to select\n* `multiplier`: magnitude of negative clamping\n\nThe procedure first discards features with retain set sparsity above `retain_threshold`, then selects the top `n_features` by forget set sparsity, and finally clamps their activations to negative `multiplier` when activated.\n\nWe quantify unlearning effectiveness through two metrics:\n\n1. **Accuracy on WMDP-bio questions:** Lower accuracy indicates successful unlearning.\n2. **Accuracy on biology-unrelated MMLU subsets:** Including High school US history, Geography, College computer science, and Human aging. Higher accuracy demonstrates preserved general capabilities.\n\nBoth metrics only evaluate on questions that the base model answers correctly across all option permutations, to reduce noise from uncertain model knowledge.\n\nWe sweep the three hyperparameters to obtain multiple evaluation results per SAE. To derive a single evaluation metric, we filter for results maintaining MMLU accuracy above 0.99 and select the minimum achieved WMDP-bio accuracy, thereby measuring optimal unlearning performance within acceptable side effect constraints.\n"
    }

    if include_example:
        example_text = "The following is an example good target benchmark and proposed solution: **benchmark: Feature absorption: Sparsity incentivizes an undesirable phenomenon called feature absorption. Imagine an SAE learned two distinct latents tracking the features \"starts with S\" and \"short\". Since \"short\" always starts with S, the SAE can increase sparsity by absorbing the \"starts with S\" feature into the \"short\" latent and then no longer needs to fire the \"starts with S\" latent when the token \"short\" is present, as it already includes the \"starts with S\" feature direction.\n\nIn general, feature absorption is incentivised any time there's a pair of concepts, A & B, where A implies B (i.e. if A activates then B will always also be active, but not necessarily the other way round). This will happen with categories/hierarchies, e.g. India => Asia, pig => mammal, red => color, etc. If the SAE learns a latent for A and a latent for B, then both will fire on inputs with A. But this is redundant–A implies B, so there's no need for the B latent to light up on A. And if the model learns a latent for A and a latent for \"B except for A\", then only one activates. This is sparser, but clearly less interpretable!\n\nFeature absorption often happens in an unpredictable manner, resulting in unusual gerrymandered features. For example, the \"starts with S\" feature may fire on 95% of tokens beginning with S, yet fail to fire on an arbitrary 5% as the \"starts with S\" feature has been absorbed for this 5% of tokens. This is an undesirable property that we would like to minimize.\n\nTo quantify feature absorption, we follow the example in Chanin et al. and use a first letter classification task. First, tokens consisting of only English letters and an optional leading space are split into a train and test set, and a supervised logistic regression probe is trained on the train set using residual stream activations from the model. This probe is used as ground truth for the feature direction in the model. Next, k-sparse probing is performed on SAE latents from the train set to find which latents are most relevant for the task. The k=1 sparse probing latent is considered as a main SAE latent for the first letter task. To account for feature splitting, as k is increased from k=n to k=n+1, if the F1 score for the k=n+1 sparse probe represents an increase of more than τ_{fs} than the F1 of the k=n probe, the k=n+1 feature is considered a feature split and is added to the set of main SAE latents performing the first letter task. We use τ_fs=0.03 in line with Chanin et al.\n\nAfter the main feature split latents for the first letter task are found, we look for test set examples where the main feature split latents fail to correctly classify the token, but the logistic regression probe is able to correctly classify the sample. We then look for a different SAE latent that fires on this sample that has a cosine similarity with the probe of at least τ_{ps}, and where the SAE latent accounts for at least τ_{pa} portion of the probe projection in the activation. We use τ_{ps}=0.025 and τ_{pa}=0.4 in line with Chanin et al.. Proposed_solution: Matryoshka SAE: Matryoshka representation learning aims to learn nested representations where lower-dimensional features are embedded within higher-dimensional ones, similar to Russian Matryoshka dolls. The key idea is to explicitly optimize for good representation at multiple scales simultaneously.\n\nWe adapt this approach to the context of sparse autoencoders. This means we can nest multiple sizes of dictionaries within each other. The largest autoencoder uses all latents for reconstruction, another uses the first half, a third uses the first quarter, and so on.\n\nThe losses of these nested autoencoders are summed. This incentivizes the initial latents to represent broadly applicable and general features, as they contribute to multiple reconstruction objectives, while later latents can focus on more specific, less frequent features."
        base_prompt_content["task_description"] = base_prompt_content["task_description"] + example_text

    return json.dumps(base_prompt_content, indent=4)

if __name__ == "__main__":
    prompt_content = generate_prompt_json(include_unlearning=True, include_example=True)
    print(prompt_content)

    # To save to a file:
    with open("prompt.json", "w") as f:
        f.write(prompt_content)