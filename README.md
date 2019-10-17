# Reinforce
Reinforce (Monte-Carlo policy gradient)   

Command to run script: 
`python reinforce.py --lr 0.0001 --gamma 0.99 --num-episodes 10 --test_frequency 200 --num_test_epsiodes 100`

#A2C

Command to run script:

`python3 a2c.py --num-episodes=40000 --lr=5e-4 --critic-lr=5e-3 --gamma=0.99 --n=50 --hidden_units=64`

#A2C_breakout

Command to run script:

`python3 a2c_breakout.py --num-episodes=40000 --lr=1e-3 --critic-lr=1e-2 --gamma=0.99 --n=120`

#A3C

Command to run script:

`python a3c_parallel.py --num-episodes=100000 --lr=1e-3 --critic-lr=1e-2 --gamma=0.99 --n=120 --num-processes=27`