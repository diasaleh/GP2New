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

import numpy as np
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()

servo_min = 150  # Min pulse length out of 4096
servo_max = 600  # Max pulse length out of 4096
servo_mid = 400
# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

print('Moving servo on channel 0, press Ctrl-C to quit...')


def drive(action):
    servo = [0]*4
    if action[0] == 0:
        servo[0] = servo_max
    else:
        servo[0] = servo_mid
    if action[1] == 0:
        servo[1] = servo_max
    else:
        servo[1] = servo_mid
    if action[2] == 0:
        servo[2] = servo_max
    else:
        servo[2] = servo_mid
    if action[3] == 0:
        servo[3] = servo_max
    else:
        servo[3] = servo_mid
    
    # Iterate through the                                                                                                                                      positions sequence 3 times.
    
    
    for i in range(4):
        pwm.set_pwm(i, 0, servo[i])
    time.sleep(.1)


action_num = 256
observation_num = 8
distance_riq = 0
RL = DeepQNetwork(n_actions=action_num,
                  n_features=observation_num,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
actionDrive = [0,0,0,0]


def convert(action):
    w=np.array(action)
    print (w)
    actionDrive = '{0:08b}'.format(int(w[0][0]))
    drive(actionDrive)
##  if action == 0:
##    actionDrive = [0,0,0,0]
##    drive(actionDrive)
##  elif action == 1:
##    actionDrive = [0,0,0,1]
##    drive(actionDrive)
##  elif action == 2:
##    actionDrive = [0,0,1,0]
##    drive(actionDrive)
##  elif action == 3:
##    actionDrive = [0,0,1,1]
##    drive(actionDrive)
##  elif action == 4:
##    actionDrive = [0,1,0,0]
##    drive(actionDrive)
##  elif action == 5:
##    actionDrive = [0,1,0,1]
##    drive(actionDrive)
##  elif action == 6:
##    actionDrive = [0,1,1,0]
##    drive(actionDrive)
##  elif action == 7:
##    actionDrive = [0,1,1,1]
##    drive(actionDrive)
##  elif action == 8:
##    actionDrive = [1,0,0,0]
##    drive(actionDrive)
##  elif action == 9:
##    actionDrive = [1,0,0,1]
##    drive(actionDrive)
##  elif action == 10:
##    actionDrive = [1,0,1,0]
##    drive(actionDrive)
##  elif action == 11:
##    actionDrive = [1,0,1,1]
##    drive(actionDrive)
##  elif action == 12:
##    actionDrive = [1,1,0,0]
##    drive(actionDrive)
##  elif action == 13:
##    actionDrive = [1,1,0,1]
##    drive(actionDrive)
##  elif action == 14:
##    actionDrive = [1,1,1,0]
##    drive(actionDrive)
##  else:
##    actionDrive = [1,1,1,1]
##    drive(actionDrive)
    return actionDrive

  #drive(actionDrive)

for i_episode in range(100):

    observation = [0,0,0,0,0,0,0,0]
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

# Done.  Terminate all signals and relax the motor.
pwm.stop()

# We have shut all our stuff down but we should do a complete
# close on all GPIO stuff.  There's only one copy of real hardware.
# We need to be polite and put it back the way we found it.
GPIO.cleanup()
