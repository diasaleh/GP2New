import numpy as np
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
import smbus
import math
from mpu6050 import mpu6050
sensor = mpu6050(0x68)

def getMotion():
    x = sensor.get_accel_data()['x']
    return x
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
    sleep(.1)

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
acc=[]
for o in range(5):
	print "hy"
	servo([0,0,0,0])
max_dis = 0
for z in range(100):
    #rd = np.random.randn(8) * 0.3
    #rd[[1,3,5,7]] = 0
    #test_angles = np.clip(s+rd, -1,1)
    print("\n\n"+str(z)+"\n\n")
    #for i in np.linspace(0,1,10):
    #    servo((1-i) * last_s + i * test_angles)
    test_angles = np.zeros(4)
    test_angles = np.random.choice([-.25,0,.25],4)
    test_angles2 = np.random.choice([-.25,0,.25],4)
   # test_angles3 = np.random.choice([-.25,0,.25],4)
    # pre_distance = distance()
    for u in range(4):
        servo(test_angles)
        servo(test_angles2)
    for t in range(10):
        acc.append(getMotion())
        #servo(test_angles3)
    acc_max = max(acc)
    # cur_distance = distance()
    #last_s = test_angles.copy()
    # div_distance = cur_distance - pre_distance
    a.append([acc_max,test_angles,test_angles2])
    # if div_distance > max_dis:
    #     max_dis = div_distance
    #     action1_max = test_angles
    #     action2_max = test_angles2

    print(acc_max)
    acc = []
    servo([0,0,0,0])   
    sleep(.5)

    #np.savetxt('newangles_m.txt', angles)
    #np.savetxt('newsenses_m.txt', senses)
a = sorted(a, key=itemgetter(0),reverse=True)
with open('outfileDelay1_acc', 'wb') as fp:
   pickle.dump(a, fp)

# with open ('outfileDelay1', 'rb') as fp:
#      a = pickle.load(fp)
# a = sorted(a, key=itemgetter(0),reverse=True)
# # with open('outfile', 'wb') as fp:
# #     pickle.dump(a, fp)
# with open ('outfile', 'rb') as fp:
#     itemlist = pickle.load(fp)


print  (a)

for pp in range(4):
    print(pp)
    sleep(5)
    for p in range(50):
	print(a[pp])
        #servo(a[len(a)-1][1])
        #servo(a[len(a)-1][2])
        #servo(a[len(a)-1][3])
        servo(a[pp][1])
        servo(a[pp][2])
        #servo(a[0][3])
sleep(3)
for pp in range(1):
    print(pp)
    sleep(5)
    for p in range(100):
        print(a[len(a)-1])
        servo(a[len(a)-1][1])
        servo(a[len(a)-1][2])
        # servo(a[len(a)-1][3])
       #  servo(a[0][1])
       #  servo(a[0][2])
       #  servo(a[0][3])
       # sleep(.1)
#angles = np.loadtxt('newangles.txt')
#senses = np.loadtxt('newsenses.txt')

