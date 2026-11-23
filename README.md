# Grid Day Types and a Learning Battery

Three kinds of machine learning on one real electricity grid, using two years (2024 and 2025) of hourly demand from Colorado's Public Service Company of Colorado (PSCO) balancing authority.

1. **Unsupervised:** k-means finds the kinds of day the grid has, checked against a no-cluster baseline and across random seeds.
2. **Supervised:** a classifier predicts tomorrow's kind of day at midnight, using only information that would be known then (no leakage).
3. **Reinforcement:** a Q-learning agent learns when a small battery should charge and discharge, scored as a share of the best possible schedule.

The notebook is [`reinforcement_learning_grid_battery_agent.ipynb`](reinforcement_learning_grid_battery_agent.ipynb). Every score is shown beside the score of a plan that learned nothing.

## Results

| Stage | Result | Floor beside it |
|---|---|---|
| Day types (k-means, k = 2) | silhouette 0.4951; identical grouping across 10 seeds (ARI 1.0) | 0.3986 on random data with no clusters |
| Midnight classifier (logistic regression) | 90.30% accuracy on 2025 | 68.70% always guessing the common type |
| Battery agent (Q-learning, predicted day types) | 83.07% of the best possible schedule on 2025 | 0% doing nothing; 80.48% a fixed clock rule; 83.29% a rule fitted to 2024 |

The agent works, but a simple rule fitted to the 2024 average day does just as well. An appendix shows that drawing the day-type boundaries with Ward clustering instead raised the agent to 86.00%.

## Data

Hourly adjusted demand from the U.S. Energy Information Administration's Form EIA-930, which is public domain. The notebook downloads EIA's six-month files for the four regions and caches them in a local `eia930_demand_2024_2025.csv.gz`. The first run downloads about 45 MB per file and takes a few minutes. EIA occasionally revises past data, so a fresh run may differ slightly in the last digits.

Source: U.S. Energy Information Administration.

## Run it

```
pip install -r requirements.txt
jupyter notebook reinforcement_learning_grid_battery_agent.ipynb
```

Built with Python, pandas, NumPy, scikit-learn, and matplotlib.
