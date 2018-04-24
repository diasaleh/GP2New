"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import RPi.GPIO as GPIO
import time

import random
from RL_brain import DeepQNetwork
action_num = 16
observation_num = 4
distance_riq = 0
RL = DeepQNetwork(n_actions=action_num,
                  n_features=observation_num,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
actionDrive = [0,0,0,0]

def drive(x):
  x = 1

def convert(action):
  print(action)
  if action == 0:
    actionDrive = [0,0,0,0]
    drive(actionDrive)
  elif action == 1:
    actionDrive = [0,0,0,1]
    drive(actionDrive)
  elif action == 2:
    actionDrive = [0,0,1,0]
    drive(actionDrive)
  elif action == 3:
    actionDrive = [0,0,1,1]
    drive(actionDrive)
  elif action == 4:
    actionDrive = [0,1,0,0]
    drive(actionDrive)
  elif action == 5:
    actionDrive = [0,1,0,1]
    drive(actionDrive)
  elif action == 6:
    actionDrive = [0,1,1,0]
    drive(actionDrive)
  elif action == 7:
    actionDrive = [0,1,1,1]
    drive(actionDrive)
  elif action == 8:
    actionDrive = [1,0,0,0]
    drive(actionDrive)
  elif action == 9:
    actionDrive = [1,0,0,1]
    drive(actionDrive)
  elif action == 10:
    actionDrive = [1,0,1,0]
    drive(actionDrive)
  elif action == 11:
    actionDrive = [1,0,1,1]
    drive(actionDrive)
  elif action == 12:
    actionDrive = [1,1,0,0]
    drive(actionDrive)
  elif action == 13:
    actionDrive = [1,1,0,1]
    drive(actionDrive)
  elif action == 14:
    actionDrive = [1,1,1,0]
    drive(actionDrive)
  else:
    actionDrive = [1,1,1,1]
    drive(actionDrive)
  return actionDrive

  #drive(actionDrive)

for i_episode in range(100):

    observation = [0,0,0,0]
    convert(observation)
    ep_r = 0
    for _ in range(4):
        predistance = 0
        action = RL.choose_action(observation)
        
        actionDrive = convert(action)

        curdistance = 1
        divdistance = curdistance - predistance
        print(actionDrive)
        observation_ = actionDrive
        

        # the smaller theta and closer to center the better
        if divdistance > distance_riq:
          reward = divdistance
        else:
          reward = -divdistance

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 100:
            RL.learn()

        observation = observation_
        total_steps += 1


RL.plot_cost()
