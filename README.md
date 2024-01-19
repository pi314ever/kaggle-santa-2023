# Santa 2023 Competition

## Setup

**Prerequisites**
- Python 3.11
- `poetry`

**Steps**
1. Clone the repository `git clone https://github.com/pi314ever/kaggle-santa-2023.git`
2. Install dependencies `poetry install`

## Algorithm Description

### A* Search with Basic Heuristic

The basic idea of this algorithm is to brute-force all combinations of moves with a simple heuristic function that optimizes for how close a state is to the solution. However, this algorithm is only feasible for small puzzles due to not pruning the search space.

Puzzles solved with this approach:

- Cube 2
- Wreath 6-12

### Deep Approximate Value Iteration

A previous approach from UCI (see [DeepCubeA](https://github.com/forestagostinelli/DeepCubeA)) used a deep neural network to approximate the value function, making it possible to solve much larger puzzles given that the puzzle had a relatively small move space.

All other puzzles were solved with this approach.

