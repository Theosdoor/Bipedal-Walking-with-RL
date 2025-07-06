# Truncated Quantile Critic + D2RL - Bipedal Walker 
Submitted as part of the degree of MSci Natural Sciences (3rd year) to the Board of Examiners in the Department of Computer Sciences, Durham University. 
This summative assignment was assessed and marked by the professor of the module in question:

| Metric | Weight | Normal | Hardcore | Difference |
|--------|--------|--------|----------|------------|
| **Core Performance** |
| Performance | (50/90) | 78% | 75% | +3% |
| Video | (15/90) | 71% | 88% | -17% |
| Sophistication | (25/90) | - | - | - |
| **Detailed Metrics** |
| Exploration | - | 91% | 84% | +7% |
| High Score | - | 78% | 86% | -8% |
| Robustness | - | 45% | 57% | -12% |
| Median | - | 83% | 70% | +13% |
| **Overall Score** | | **78%** | **75%** | **+3%** |

## Grade: 1st - 80%.
> "*Normal environment performance:* The normal agent has near perfect exploration in the early learning phase and it has an excellent highscore (best 90% quantile over 100 steps) with a value of 330.21. It has an outstanding median area under the overall learning curve. Overall, the normal agent has an excellent performance.

> *Normal environment video:* The normal agent has learned an excellent policy with a score of 330.08 at episode 179. It has learned a very fast and efficient policy with just a little room left for perfection.

> *Hardcore environment performance:* The hardcore agent has outstanding exploration in the early learning phase and it has an outstanding highscore (best 90% quantile over 100 steps) with a value of 325.9. It has an excellent median area under the overall learning curve. Overall, the hardcore agent has an excellent performance.

> *Hardcore environment video:* The hardcore agent has learned an outstanding policy with a score of 325.98 at episode 959, ranking in the top 10% of the class. It has learned a very fast and efficient policy that handles the obstacles very well and hast just a little room left for perfection.

> *Sophistication and mastery of the domain:* This is an excellent report presenting a sophisticated TQCD2RL agent enhanced with a hybrid ERE+PER buffer and tuned via Bayesian optimisation. The theoretical grounding is solid throughout, with a detailed explanation of how the agent addresses overestimation bias and sample efficiency, and a good understanding of the practical trade-offs in using ERE and PER together. The experimental design is thoughtful, including both hyperparameter sweeps and ablation studies that support the effectiveness of the proposed approach. The PER ablation is well contextualised, though further exploration of the joint scheduling and tuning of ERE and PER could improve the scientific rigour. The reward shaping and practical considerations (e.g. reward clipping and buffer sizing) demonstrate a strong grasp of implementation subtleties. This shows clear mastery of the domain with some originality and independent thinking." - Dr Robert Lieck

## Abstract:
This paper proposes using Truncated Quantile Critics (TQC) with D2RL architecture and mixed-strategy buffer comprising Emphasising Recent Experiences (ERE) and Prioritised Experience Replay (PER). It found promising hyperparameters through a Bayesian search, and the resulting agent is able to solve both BipedalWalker and BipedalWalkerHardcore.

### Video of Agent:
  >![Gifdemo](https://github.com/Theosdoor/Bipedal-Walker-with-TQC-and-ERE-PER/blob/main/hardcore_env/hardcore-video,episode=959,score=326.gif)

## Contents:
* hardcore_env / normal_env - for normal and challenging hardcore environments
  * env-agent-log.txt - rewards tracked over episodes.
  * env-agent.ipynb - Implemented TQC+D2RL with an ERE+PER buffer. Each env has different hyperparameters.
  * env-video,episode=xxx,score=xxx.gif - Video of the agent completing the bipedal normal/hardocore environment
* agent_paper.pdf - Paper outlining methodology

## Results:
> These graphs show my individualised convergence data relative to everyone else for the normal (left) and hardcore (right) environment. The top row shows my learning progress over time, including the median curve and the 10% and 90% quantiles. The bottom row shows my agent’s profile along the different dimensions. The colour in all graphs relates to the final performance grade.
  > ![Performance Graphs](performance_data.png?raw=true "Performance Graphs")


  

