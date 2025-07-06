# Solving BipedalWalker with Truncated Quantile Critics (TQC) + D2RL + ERE+PER Buffer

## Overview
Submitted as part of 3rd year MSci Natural Sciences to the Department of Computer Science at Durham University. It was assessed by module leader Dr Robert Lieck.

- **Grade:** 1st Class (80%)
- **Overall Scores:** 78% (Normal), 75% (Hardcore)
- **Notable Rankings:** Top 10% of class for Hardcore environment performance

### Marking Feedback
> "*Normal environment performance:* The normal agent has near perfect exploration in the early learning phase and it has an excellent highscore (best 90% quantile over 100 steps) with a value of 330.21. It has an outstanding median area under the overall learning curve. Overall, the normal agent has an excellent performance.

> *Normal environment video:* The normal agent has learned an excellent policy with a score of 330.08 at episode 179. It has learned a very fast and efficient policy with just a little room left for perfection.

> *Hardcore environment performance:* The hardcore agent has outstanding exploration in the early learning phase and it has an outstanding highscore (best 90% quantile over 100 steps) with a value of 325.9. It has an excellent median area under the overall learning curve. Overall, the hardcore agent has an excellent performance.

> *Hardcore environment video:* The hardcore agent has learned an outstanding policy with a score of 325.98 at episode 959, ranking in the top 10% of the class. It has learned a very fast and efficient policy that handles the obstacles very well and hast just a little room left for perfection.

> *Sophistication and mastery of the domain:* This is an excellent report presenting a sophisticated TQCD2RL agent enhanced with a hybrid ERE+PER buffer and tuned via Bayesian optimisation. The theoretical grounding is solid throughout, with a detailed explanation of how the agent addresses overestimation bias and sample efficiency, and a good understanding of the practical trade-offs in using ERE and PER together. The experimental design is thoughtful, including both hyperparameter sweeps and ablation studies that support the effectiveness of the proposed approach. The PER ablation is well contextualised, though further exploration of the joint scheduling and tuning of ERE and PER could improve the scientific rigour. The reward shaping and practical considerations (e.g. reward clipping and buffer sizing) demonstrate a strong grasp of implementation subtleties. This shows clear mastery of the domain with some originality and independent thinking."
— Dr. Robert Lieck, Durham University

## Abstract
This project investigates combining Truncated Quantile Critics (TQC) with D2RL architecture and a mixed-strategy experience replay buffer of both Emphasising Recent Experiences (ERE) and Prioritised Experience Replay (PER). Through Bayesian hyperparameter optimization, the resulting agent successfuly solves both the standard and hardcore variants of the BipedalWalker environment.

## Performance Results

### Quantitative Results

| Metric | Weight | Normal | Hardcore |
|--------|--------|--------|----------|
| **Core Performance** |
| Performance | (50/90) | 78% | 75%
| Video | (15/90) | 71% | 88%
| Sophistication | (25/90) | 78% | 78%
| **Detailed Metrics** |
| Exploration | - | 91% | 84%
| High Score | - | 78% | 86%
| Robustness | - | 45% | 57%
| Median | - | 83% | 70%
| **Overall Score** | | **78%** | **75%**

### Agent Video
![Hardcore Agent Demo](https://github.com/Theosdoor/Bipedal-Walker-with-TQC-and-ERE-PER/blob/main/hardcore_env/hardcore-video,episode=959,score=326.gif)

*The agent successfully navigating the hardcore environment with obstacles, achieving a score of 326 at episode 959.*

## Technical Approach

1. **Truncated Quantile Critics (TQC)**: Addresses overestimation bias in Q-learning
2. **D2RL Architecture**: Deep dense residual learning for improved gradient flow
3. **Hybrid Buffer Strategy**: 
   - **ERE**: Emphasizes recent experiences for faster adaptation
   - **PER**: Prioritizes important transitions for efficient learning
4. **Bayesian Optimization**: Systematic hyperparameter tuning for optimal performance

## Repo Structure

```
├── hardcore_env / normal_env   # Normal and hardocre environment implementations
│   ├── env-agent-log.txt       # Training rewards and episode data
│   ├── env-agent.ipynb         # TQC+D2RL implementation with ERE+PER buffer. Each env has different hyperparameters.
│   └── env-video,episode=959,score=326.gif  # Demo videos
├── agent_paper.pdf             # Detailed methodology and results
├── performance_data.png        # Comparative performance visualizations
└── README.md                   # This file
```

## Results
> These graphs show my individualised convergence data relative to everyone else for the normal (left) and hardcore (right) environment. The top row shows my learning progress over time, including the median curve and the 10% and 90% quantiles. The bottom row shows my agent’s profile along the different dimensions. The colour in all graphs relates to the final performance grade.
  > ![Performance Graphs](performance_data.png?raw=true "Performance Graphs")

