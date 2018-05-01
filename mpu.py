import numpy as np
# from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D
from keras import backend as K
import RPi.GPIO as GPIO
import time

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
def getMotion():
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
    
    return beschleunigung_xout,beschleunigung_yout,beschleunigung_zout,gyroskop_xout,gyroskop_yout,gyroskop_zout,x_rot,y_rot
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
    action = list(map(scaling,action))
    #print (action)

    for i in range(8):
        pwm.set_pwm(i, 0,int( action[i]))


action_num = 256
observation_num = 8
distance_riq = 0


def sleep(x):
    time.sleep(x)



def clear_output(wait):
    i = 1
    if i == 0:
        i+=1

    
angles = []
senses = []
s = np.array([1,0,1,0,1,0,1,0], dtype=np.float64)
last_s = s.copy()
servo(s)
sleep(1)

for z in range(0):
    print("hi")
    rd = np.random.randn(8) * 0.3
    rd[[1,3,5,7]] = 0
    test_angles = np.clip(s+rd, -1,1)

    for i in np.linspace(0,1,10):
        servo((1-i) * last_s + i * test_angles)

    last_s = test_angles.copy()
    sleep(.5)

    sense_sum = np.zeros(8)
    for i  in range(10):
        sense_sum += np.array(getMotion())
        sleep(.05)

    sense = sense_sum/10.0

    angles.append(test_angles.copy())
    senses.append(sense)

    clear_output(wait=True)
    a = np.array(senses)
    #fig = plt.plot(a[:,0], c = 'r')
    #fig = plt.plot(a[:,1], c = 'b')
    #plt.show()

    np.savetxt('angles.txt', angles)
    np.savetxt('senses.txt', senses)
angles = np.loadtxt('angles.txt')
senses = np.loadtxt('senses.txt')
axis = 0
u = np.mean(senses, axis = 0)
std = np.std(senses, axis = 0)

senses_norm = (senses - u)/std

single_axis = np.array(senses_norm)[:,axis]
#fig = plt.plot(single_axis)

def line_fit(single_axis, joint_states):
    m, b = np.polyfit(single_axis, joint_states, 1)
    
    T = np.linspace(np.min(single_axis), np.max(single_axis))

    # fig = plt.scatter(single_axis, joint_states)
    # fig = plt.plot(T, [m*t + b for t in T])

    return m, b

line_fit(single_axis, np.array(angles)[:,6])
coeff = [line_fit(single_axis, np.array(angles)[:,j]) for j in range(8)]

T = np.linspace(np.min(single_axis)-2, np.max(single_axis)+2, 20)

for desired_h in T:
    js = np.array([c[0]*desired_h+c[1] for c in coeff])
    servo(js)
    sleep(.1)

#multi input
model = Sequential()
model.add(Dense(64, input_dim=8)) #two for xy
model.add(Activation('tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.20))
model.add(Dense(8))
model.add(Activation('tanh'))

#for a mean squared error regression problem
model.compile(optimizer='rmsprop',loss='mse')

#model training

axis_vis = senses_norm[:,0]

T = np.linspace(np.min(axis_vis), np.max(axis_vis), 30)

for i in range(20):
    model.fit(senses_norm[:,:], np.array(angles), verbose=False, epochs=20)

    joint_angles = []
    for m in T:
        joint_angles.append(model.predict(np.array([[m, 0]]))[0])

    clear_output(wait=True)

   # fig = plt.plot(T, joint_angles)
    #plt.show()

def tilt(x,y):
    joint_angles = model.predict(mp.array([[x, y]]))[0]
    servo(joint_angles)
    sleep(.1)

T = np.linspace(0, np.pi*2.0 , 20)

for i in range(30):
    print(str(1))
    print(str(i))
    sense_n = np.zeros(8)
    for i  in range(10):
        sense_n += np.array(getMotion())
        sleep(.05)
    #sense_n = sense_n/10.0
    print(sense_n)
    u = np.mean(sense_n)
    std = np.std(sense_n)
    sense_no = (sense_n - u)/std
    print(sense_no)
    for t in T:
        joint_angles = model.predict(np.array([[-sense_no, -sense_no]]))[0]
        servo(joint_angles)
        sleep(.01)

for i in range(30):
    print(str(2))
    print(str(i))
    for t in T:
    	sense_n = np.zeros(8)
    	for i  in range(10):
        	sense_n += np.array(getMotion()) 
        	sleep(.05)
    	#sense_n = sense_n/10.0
    	u = np.mean(sense_n)
    	std = np.std(sense_n)
    	sense_no = (sense_n - u)/std
        joint_angles = model.predict(np.array([[-sense_no[0], -sense_no[1]]]))[0]
        servo(joint_angles)
        sleep(.01)

for i in range(30):
    print(str(i))
    sense_n = np.zeros(8)
    for i  in range(10):
        sense_n += np.array(getMotion())
        sleep(.05)
    sense_n = sense_n/10.0
    for t in T:
        joint_angles = model.predict(np.array([[sense_n[0], sense_n[1]]]))[0]
        servo(joint_angles)
        sleep(.01)

for i in range(30):
    print(str(i))
    for t in T:
	sense_n = np.zeros(8)
    	for i  in range(10):
        	sense_n += np.array(getMotion())
        	sleep(.05)
	sense_n = sense_n/10.0
        joint_angles = model.predict(np.array([[sense_n[0], sense_n[1]]]))[0]
        servo(joint_angles)
        sleep(.01)

for i in range(30):
    print(str(i))
    for t in T:
        joint_angles = model.predict(np.array([[100, 0]]))[0]
        servo(joint_angles)
        sleep(.1)


joint_angles = model.predict(np.array([[0, 0]]))[0]
servo(joint_angles)



