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
#set GPIO Pins
GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 18
GPIO_ECHO = 24
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
 
def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance
 
 
pwm = Adafruit_PCA9685.PCA9685()

servo_min = 150  # Min pulse length out of 4096
servo_max = 500  # Max pulse length out of 4096
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
    servo = [0]*8
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
    if action[4] == 0:
        servo[4] = servo_max
    else:
        servo[4] = servo_mid
    if action[5] == 0:
        servo[5] = servo_max
    else:
        servo[5] = servo_mid
    if action[6] == 0:
        servo[6] = servo_max
    else:
        servo[6] = servo_mid
    if action[7] == 0:
        servo[7] = servo_max
    else:
        servo[7] = servo_mid
    # Iterate through the                                                                                                                                      positions sequence 3 times.
    
    
    for i in range(8):
        pwm.set_pwm(i, 0, servo[i])
   	time.sleep(.1)


action_num = 2
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
    
    actionDrive = '{0:08b}'.format(action)
    actionDrive = list(actionDrive)
    
    drive(map(int,actionDrive))
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
    ep_r = 0
    for _ in range(4):
        predistance = distance()
        action = RL.choose_action(observation)
        
        actionDrive = convert(action)

        curdistance = distance()
        divdistance = curdistance - predistance
        print (divdistance)
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

while True:
	action = RL.forward_action(observation)
	actionDrive = convert(action)
	observation_ = actionDrive
	observation = observation_
RL.plot_cost()

# Done.  Terminate all signals and relax the motor.
pwm.stop()

# We have shut all our stuff down but we should do a complete
# close on all GPIO stuff.  There's only one copy of real hardware.
# We need to be polite and put it back the way we found it.
GPIO.cleanup()
