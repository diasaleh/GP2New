import numpy as np
# from matplotlib import pyplot as plt
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D
#from keras import backend as K
import RPi.GPIO as GPIO
import time
import operator
import pickle

import collections
from operator import itemgetter
import random
from RL_brain import DeepQNetwork

import numpy as np
import Adafruit_PCA9685

#set GPIO Pins
import smbus
import math
bus = smbus.SMBus(1) # bus = smbus.SMBus(0) fuer Revision 1
address = 0x68       # via i2cdetect
def scaling(x):
    OldMax = 1
    OldMin = -1
    NewMax = 600
    NewMin = 150
    OldValue = x
    OldRange = (OldMax - OldMin)
    if (OldRange == 0):
        NewValue = NewMin
    else:
        NewRange = (NewMax - NewMin)  
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return (NewValue)
 
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
def ggetMotion():
    bus.write_byte_data(address, power_mgmt_1, 0)
 
    # print "Gyroskop"
    # print "--------"
 
    gyroskop_xout = read_word_2c(0x43)
    gyroskop_yout = read_word_2c(0x45)
    gyroskop_zout = read_word_2c(0x47)
 
    # print "gyroskop_xout: ", ("%5d" % gyroskop_xout), " skaliert: ", (gyroskop_xout / 131)
    # print "gyroskop_yout: ", ("%5d" % gyroskop_yout), " skaliert: ", (gyroskop_yout / 131)
    # print "gyroskop_zout: ", ("%5d" % gyroskop_zout), " skaliert: ", (gyroskop_zout / 131)
    # print "Beschleunigungssensor"
    # print "---------------------"
 
    beschleunigung_xout = read_word_2c(0x3b)
    beschleunigung_yout = read_word_2c(0x3d)
    beschleunigung_zout = read_word_2c(0x3f)
 
    beschleunigung_xout_skaliert = beschleunigung_xout / 16384.0
    beschleunigung_yout_skaliert = beschleunigung_yout / 16384.0
    beschleunigung_zout_skaliert = beschleunigung_zout / 16384.0
 
    # print "beschleunigung_xout: ", ("%6d" % beschleunigung_xout), " skaliert: ", beschleunigung_xout_skaliert
    # print "beschleunigung_yout: ", ("%6d" % beschleunigung_yout), " skaliert: ", beschleunigung_yout_skaliert
    # print "beschleunigung_zout: ", ("%6d" % beschleunigung_zout), " skaliert: ", beschleunigung_zout_skaliert
 
    x_rot =  get_x_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert)
    y_rot =  get_y_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert)
    
    #return x_rot,y_rot
    return beschleunigung_xout,beschleunigung_yout,beschleunigung_zout,gyroskop_xout,gyroskop_yout,gyroskop_zout,x_rot,y_rot
# return beschleunigung_xout,beschleunigung_yout,beschleunigung_zout,gyroskop_xout,gyroskop_yout,gyroskop_zout,x_rot,y_rot
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

def servo(action):
    print (action)
    action = list(map(scaling,action))
    for i in range(4):
        pwm.set_pwm(i, 0,int( action[i]))
    sleep(.5)

action_num = 256
observation_num = 8
distance_riq = 0


def sleep(x):
    time.sleep(x)
    
angles = []
senses = []
s = np.array([1,0,1,0], dtype=np.float64)
last_s = s.copy()
sleep(1)
a=[]
for o in range(5):
	print "hy"
	servo([0,0,0,0])
max_dis = 0
for z in range(20):
    #rd = np.random.randn(8) * 0.3
    #rd[[1,3,5,7]] = 0
    #test_angles = np.clip(s+rd, -1,1)

    #for i in np.linspace(0,1,10):
    #    servo((1-i) * last_s + i * test_angles)
    test_angles = np.zeros(4)
    test_angles = np.random.choice([-.25,0,.25],4)
    test_angles2 = np.random.choice([-.25,0,.25],4)
    pre_distance = distance()
    for u in range(4):
        servo(test_angles)
        servo(test_angles2)
    cur_distance = distance()
    #last_s = test_angles.copy()
    div_distance = cur_distance - pre_distance
    a.append([div_distance,test_angles,test_angles2])
    # if div_distance > max_dis:
    #     max_dis = div_distance
    #     action1_max = test_angles
    #     action2_max = test_angles2

    sleep(.5)
    print(cur_distance)
    servo([0,0,0,0])   

    #np.savetxt('newangles_m.txt', angles)
    #np.savetxt('newsenses_m.txt', senses)
a = sorted(a, key=itemgetter(0),reverse=True)
with open('outfile', 'wb') as fp:
    pickle.dump(a, fp)
# with open ('outfile', 'rb') as fp:
#     itemlist = pickle.load(fp)
for pp in range(5):
    for p in range(50):
        servo(a[pp][1])
        servo(a[pp][2])
        sleep(.2)

#angles = np.loadtxt('newangles.txt')
#senses = np.loadtxt('newsenses.txt')

