# Connect4 zero


Univeristy Project: [In-depth study on Reinforcement Learning (RL)](https://github.com/gianlucagiudice/alpha-zero-general/blob/master/In%20Depth%20Study%20-%20Reinforcement%20Learning.pdf) by **Gianluca Giudice - 830694**



### Version
- `python3.8`

### Install libraries
```
pip install -r requirements.txt
```
### Structure

#### Root structure
```
.
├── README.md
├── analyze_training.ipynb (analyze training results)
├── connect4_human-vs-random.py (play against a random agent)
├── connect4_human-vs-rl.py (play against the RL agent)
├── doc (papers useful for report)
├── logs (training logs)
├── models (keras model)
├── report (In-depth study)
├── src (source code)
├── temp (temp keras checkpoint used for training)
├── train_connect4.py (train the model)
```
#### Code structure
```
.
├── Arena.py (Agents play against each other)
├── Coach.py (Lerner)
├── Game.py (Abstract definition of a Game)
├── MCTS.py (Monte Carlo Tree Search class)
├── NeuralNet.py (Neural network abstract class)
├── Player.py (Different types of agents)
├── connect4 (connect4 game)
│   ├── Connect4Board.py (connect4 board)
│   ├── Connect4Game.py (connect4 logic)
│   ├── Connect4Players.py (Agents in connect4)
│   └── keras  (neural network used for connect4)
│       ├── Connect4NNet.py
│       ├── NNet.py ()
└── utils.py
```
### Instructions
#### To train the model
```
./python train_connect4.py
```

#### To play against the agent
```
./python connect4_human-vs-rl.py
```
### Output
- **Model**: The best model will be saved in `temp/best.h5`
- **Log**: Training logs in `logs/`
