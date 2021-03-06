
""" Deep Q network, Using: Tensorflow: 1.0 gym: 0.7.3 """
import RPi.GPIO as GPIO
import time

import random
from RL_brain import DeepQNetwork
from mpu6050 import mpu6050
import numpy as np
import Adafruit_PCA9685

#set GPIO Pins
import smbus
import math
bus = smbus.SMBus(1) # bus = smbus.SMBus(0) fuer Revision 1
address = 0x68       # via i2cdetect
sensor = mpu6050(0x68)
# Register
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c
difreq =300 
def read_byte(reg):
    return bus.read_byte_data(address, reg)
 
def read_word(reg):
    h = bus.read_byte_data(address, reg)
    l = bus.read_byte_data(address, reg+1)
    value = (h << 8) + l
    return value
 
def read_word_2c(reg):
    val = read_word(reg)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val
 
def dist(a,b):
    return math.sqrt((a*a)+(b*b))
 
def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)
 
def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)
def getMotion2():
	bus.write_byte_data(address, power_mgmt_1, 0)
 
	#print "Gyroskop"
	#print "--------"
 
	gyroskop_xout = read_word_2c(0x43)
	gyroskop_yout = read_word_2c(0x45)
	gyroskop_zout = read_word_2c(0x47)
 
	#print "gyroskop_xout: ", ("%5d" % gyroskop_xout), " skaliert: ", (gyroskop_xout / 131)
	#print "gyroskop_yout: ", ("%5d" % gyroskop_yout), " skaliert: ", (gyroskop_yout / 131)
	#print "gyroskop_zout: ", ("%5d" % gyroskop_zout), " skaliert: ", (gyroskop_zout / 131)
	#print "Beschleunigungssensor"
	#print "---------------------"
 
	beschleunigung_xout = read_word_2c(0x3b)
	beschleunigung_yout = read_word_2c(0x3d)
	beschleunigung_zout = read_word_2c(0x3f)
 
	beschleunigung_xout_skaliert = beschleunigung_xout / 16384.0
	beschleunigung_yout_skaliert = beschleunigung_yout / 16384.0
	beschleunigung_zout_skaliert = beschleunigung_zout / 16384.0
 
	print "beschleunigung_xout: ", ("%6d" % beschleunigung_xout), " skaliert: ", beschleunigung_xout_skaliert
	#print "beschleunigung_yout: ", ("%6d" % beschleunigung_yout), " skaliert: ", beschleunigung_yout_skaliert
	#print "beschleunigung_zout: ", ("%6d" % beschleunigung_zout), " skaliert: ", beschleunigung_zout_skaliert
 
	#print "X Rotation: " , get_x_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert)
	#print "Y Rotation: " , get_y_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert)
	return beschleunigung_xout
def getMotion():
	x = sensor.get_accel_data()['x']
	print(x)
	return x
GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 18
GPIO_ECHO = 24
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
def sleep(x):
    time.sleep(x)
 
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

servo_min = 300  # Min pulse length out of 4096
servo_max = 420  # Max pulse length out of 4096
servo_mid = 320
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
    #if action[4] == 0:
    #    servo[4] = servo_max
    #else:
    #    servo[4] = servo_mid
    #if action[5] == 0:
    #    servo[5] = servo_max
    #else:
    #    servo[5] = servo_mid
    #if action[6] == 0:
    #    servo[6] = servo_max
    #else:
    #    servo[6] = servo_mid
    #if action[7] == 0:
    #    servo[7] = servo_max
    #else:
    #    servo[7] = servo_mid
    # Iterate through the                                                                                                                                      positions sequence 3 times.
    
    
    for i in range(4):
        pwm.set_pwm(i, 0, servo[i])


action_num = 16
observation_num = 4
distance_riq = 1
RL = DeepQNetwork(n_actions=action_num,
                  n_features=observation_num,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
actionDrive = [0,0,0,0]


def convert(action):
    
    actionDrive = '{0:04b}'.format(action)
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
#senses = []
  #drive(actionDrive)
#obb = np.int64(16)
#convert(obb)
observation = [0,0,0,0]
#sleep(0.5)
#for _ in range(10):
#    for i  in range(10):
#        senses.append( getMotion())
#        sleep(.05)
#max_stable = np.max(senses)
#print("\n\n"+str(max_stable))
#sleep(5)
for i in range(4):
       pwm.set_pwm(i, 0, 375)
sleep(2)
for i in range(4):
       pwm.set_pwm(i, 0, 450)
sleep(2)
for i_episode in range(100):
    print("\n i = "+str(i_episode))
    #convert(observation)
    ep_r = 0
    obb = np.int64(16)
    #action = RL.choose_action([0,0,0,0,0,0,0,0])
    #action2 = RL.choose_action(action)

    for _ in range(4):
     	predistance = distance()
     	print("pre destance = " +str(predistance))
     #    preacc = getMotion()
	    # curacc = getMotion()
	action = RL.choose_action(observation)
	actionDrive = convert(action)
	sleep(.5)
     	curdistance = distance()
     	print("destance = " +str( curdistance))
     	divdistance = curdistance - predistance
	    # #difacc = curacc-preacc

        #actionDrive = convert(action)
	#senses = []
    	#for i  in range(10):
        #	senses.append( getMotion())
        #	sleep(.05)
	#max_sense = np.max(senses)

        #actionDrive = convert(action2)

        observation_ = actionDrive
        #print ("\n\n sense = "+str(max_sense))
   
	#sleep(.5)
        #r = sense - max_stable

        # the smaller theta and closer to center the better
        if divdistance > distance_riq :
          reward = divdistance*100
	  print("\n Reward + "+str(reward))
        else:
          reward = -100

#        if sense > 300:
#	    print("Reward300 + ")
 #           RL.store_transition(observation, action, reward, observation_)
  #          RL.store_transition(observation, action, reward, observation_)
   #         RL.store_transition(observation, action, reward, observation_)

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 10:
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
