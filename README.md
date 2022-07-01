# Pareto-Weighted-Sum-Tuning
The Pareto-Weighted-Sum-Tuning (PWST) utilizes Learning-to-Rank Machine Learning to help solve Pareto (Multiobjective, Multicriteria) Optimization Problems. This codebase utilizes PWST to run experiments with a sample user/decision-maker on an example stock-pricing dataset. PWST was developed by Harry Wang at the University of Michigan.

Pareto-Weighted-Sum-Tuning was accepted and presented at the 2020 International Conference on Machine Learning, Computational Optimization, and Data Science (LOD) (https://lod2020.icas.xyz/program/).

LOD 2020 Slides:
https://docs.google.com/presentation/d/1WGXAiDIJQ18kT1rCZL_hl6W0rJk9UgFT4g3nL_Fu-AM/edit?usp=sharing

Publication Link:
https://www.springerprofessional.de/en/pareto-weighted-sum-tuning-learning-to-rank-for-pareto-optimizat/18742156

# Pareto-Weighted-Sum-Tuning Abstract
The weighted-sum method is a commonly used technique in Multi-objective optimization to represent different criteria considered in a decision-making and optimization problem. Weights are assigned to different criteria depending on the degree of importance. However, even if decision-makers have an intuitive sense of how important each criteria is, explicitly quantifying and hand-tuning these weights can be difficult. To address this problem, we propose the Pareto-Weighted-Sum-Tuning algorithm as an automated and systematic way of trading-off between different criteria in the weight-tuning process. Pareto-Weighted-Sum-Tuning is a configurable online-learning algorithm that uses sequential discrete choices by a decision-maker on sequential decisions, eliminating the need to score items or weights. We prove that utilizing our online-learning approach is computationally less expensive than batch-learning, where all the data is available in advance. Our experiments show that Pareto-Weighted-Sum-Tuning is able to achieve low relative error with different configurations.

# Usage
To run a sample PWST experiment referenced, run the following command. Graphs will be generated to display the results produced by PWST.
```bash
python example.py
```
