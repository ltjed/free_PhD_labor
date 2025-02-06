1) Run 1 (competition_threshold=0.02, orthogonality_penalty=0.01):
   - Use the existing "CustomTrainer" but add a new term: λ_2 * Σ(c_ij * |f_i^T f_j|)
   - Set competition_threshold=0.02 to identify strongly overlapping features
   - Evaluate with “python experiment.py --out_dir=run_1”

2) Run 2 (competition_threshold=0.02, orthogonality_penalty=0.02):
   - Increase orthogonality penalty to see if stronger constraints improve interpretability
   - Evaluate with “python experiment.py --out_dir=run_2”

3) Run 3 (competition_threshold=0.05, orthogonality_penalty=0.01):
   - Evaluate higher threshold for competition mask, so fewer features are considered “competing”
   - Evaluate with “python experiment.py --out_dir=run_3”

4) Run 4 (competition_threshold=0.05, orthogonality_penalty=0.02):
   - Combine higher threshold with stronger penalty
   - Evaluate with “python experiment.py --out_dir=run_4”

5) Run 5 (competition_threshold=0.02, orthogonality_penalty=0.025):
   - Explore an even stronger penalty, to see if interpretability is improved or if it degrades reconstruction
   - Evaluate with “python experiment.py --out_dir=run_5”

6) Run 6 (additional baseline comparison if needed):
   - Possibly revert to threshold=0.02, penalty=0.0 (no orthogonality) for an additional control
   - Evaluate with “python experiment.py --out_dir=run_6”

We can add or skip additional runs as results warrant.
