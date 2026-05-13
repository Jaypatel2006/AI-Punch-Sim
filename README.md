# AI Based Punching Simulator
An AI-powered boxing training and simulation system that combines Computer Vision, Reinforcement Learning, and Game Development to create an interactive virtual punching trainer.

## Overview
The AI Based Punching Simulator is a real-time boxing trainer that uses a webcam to track body movements and detect punches. The system evaluates user performance such as:
- Punch Speed
- Accuracy
- Timing
- Form and Technique

It also simulates a virtual boxing environment where the player interacts with an AI opponent.

The project aims to provide an affordable and accessible boxing trainer using only:
- A webcam
- A computer
- AI-based pose estimation

## Objectives
- Real-time pose tracking using computer vision
- Punch detection and classification
- AI opponent simulation
- Performance analysis and scoring
- Interactive feedback and improvement suggestions

## Features
### Pose Tracking
Extracts body keypoints in real time:
- Wrists
- Elbows
- Shoulders
- Torso

### Punch Detection
Detects and classifies punches such as:
- Jab
- Cross
- Hook
- Uppercut

### AI Opponent
Uses Reinforcement Learning concepts to simulate intelligent responses.

### Game Simulation
Built using Pygame to create a virtual boxing arena.

### Performance Analytics
Analyzes:
- Punch speed
- Accuracy
- Stamina
- Timing
- Technique

## Technologies used
- Python
- OpenCV
- MediaPipe / Pose Estimation
- Pygame
- NumPy
- Reinforcement Learning
- Q-Learning
- MDP (Markov Decision Process)

## AI Concepts Used
### Markov Decision Process (MDP)
Used for modeling:
- States
- Actions
- Rewards
- Decision-making process
### Q-Learning
The AI opponent improves over time using reward-based learning.
Q-value update equation:
```text
Q(s,a) ← Q(s,a) + α[r + γmaxQ(s′,a′) − Q(s,a)]
```

## Assumptions
- User stands in front of webcam
- Proper lighting conditions
- Single user interaction
- Hands remain visible 

## Project Structure
```text
AI-Punch-Sim/
│
├── block.py
├── game.py
├── main.py
├── requirements.txt
├── steps.txt
└── README.md
```

## How to Run
### 1. Clone Repository
```bash
git clone https://github.com/Jaypatel2006/AI-Punch-Sim.git
cd AI-Punch-Sim
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Project
```bash
python game.py
python main.py
```
## Learning Outcomes
- Reinforcement Learning concepts
- Real-time Computer Vision
- AI-based game simulation
- Debugging and optimization
- Interactive system development
