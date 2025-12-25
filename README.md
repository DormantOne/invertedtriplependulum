🔥 Pendulum Furnace
Evolving liquid neural networks to balance a triple pendulum.

What It Does
Uses evolutionary search to find the right hyperparameters (reservoir size, leak rates, spectral radius) and reward weights for a recurrent neural network that can balance a chaotic triple pendulum indefinitely.

Run It
bashpip install torch numpy flask
python pendulum_furnace_v22.py
Open http://127.0.0.1:5000

The Breakthrough
Stuck at 150 steps until we added a constraint: the agent must allocate at least 15% of its reward to "staying alive." Evolution then discovered 60% is optimal → 900 steps (max).
Architecture

Controller: Multi-timescale echo state network (~900 neurons)
Training: Actor-critic per individual
Evolution: Adaptive per-gene mutation rates (sensitive genes get smaller steps)
Physics: Featherstone spatial algebra, 500Hz simulation

Files
pendulum_furnace_v22.py — the version that achieved 900 steps
*_best_v22.json — saved champion genome
*_best_v22.pt — saved champion weights

License
MIT
