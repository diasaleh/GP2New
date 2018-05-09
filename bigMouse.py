import array
import sys
import struct
from gopigo import *
import usb.core
import usb.util
import numpy as np
import RPi.GPIO as GPIO
import time
import operator
import pickle
import collections
from operator import itemgetter
import random
import numpy as np
import Adafruit_PCA9685
import smbus
import math
import threading
file = open( "/dev/input/mice", "rb" );
speed=150
learning_episodes = 0

exitFlag = [0]*learning_episodes
results = [None] * learning_episodes
debug = 0
def getMouseEvent():
    buf = file.read(3)
    button = ord( buf[0] )
    bLeft = button & 0x1
    bMiddle = ( button & 0x4 ) > 0
    bRight = ( button & 0x2 ) > 0
    x,y = struct.unpack( "bb", buf[1:] )
    if debug:
        print ("L:%d, M: %d, R: %d, x: %d, y: %d\n" % (bLeft,bMiddle,bRight, x, y) )
    return [bLeft,bMiddle,bRight,x,y]
    

flag=0
set_speed(speed)

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

def getMouseData(idd,results):

	a=[0]*3
	print (threading.currentThread().getName(), 'Starting '+str(idd))
	while( not exitFlag[idd] ):
        	[l,m,r,x,y]=getMouseEvent() #Get the inputs from the mouse
        # if debug:
        #     print l,m,r,x,y
        # print x,"\t",y

        #If there is a signinficant mouse movement Up (positive y-axis)
        	if y >20:
            		print("fwd()")  #Move forward
            		a[0] = a[0]+y
        #If there is a signinficant mouse movement Down (negative y-axis)
        	elif y<-20:
            		print("bwd()")  #Move Back
            		a[0] = a[0]+y
        #If there is a signinficant mouse movement Left (positive x-axis)
        	elif x<-20:
            		print("left()") #Move left
           	        a[1] = a[1]+1
        #If there is a signinficant mouse movement Right (negative x-axis)
        	elif x>20:
            		print("right()")    #Move Right
            		a[2] = a[2]+1
        	time.sleep(.01)
		results[idd] = a
	print (threading.currentThread().getName(), 'Exiting '+str(idd))
	# print ("baigiau! :)")
 
 
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
    

def servoControl(idd,test_angles,test_angles2):
    print (threading.currentThread().getName(), 'Starting '+str(idd))
    for u in range(4):
	servo(test_angles)
    	servo(test_angles2)
    print (threading.currentThread().getName(), 'Exiting '+str(idd))
    exitFlag[idd] = 1
    
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
sum_array=[]

for i in range(learning_episodes):
    
    #rd = np.random.randn(8) * 0.3
    #rd[[1,3,5,7]] = 0
    #test_angles = np.clip(s+rd, -1,1)
    print("\n\n"+str(i)+"\n\n")
    #for i in np.linspace(0,1,10):
    #    servo((1-i) * last_s + i * test_angles)
    test_angles = np.zeros(4)
    test_angles2 = np.zeros(4)
    test_angles = np.random.choice([-.25,0,.25],4)
    for v in range(4):
        test_angles2[v]=np.random.choice([-.25,0,.25],1)
        while test_angles2[v] == test_angles[v]:
            test_angles2[v]=np.random.choice([-.25,0,.25],1)
    #test_angles2 = np.random.choice([-.25,0,.25],4)
   # test_angles3 = np.random.choice([-.25,0,.25],4)
    # pre_distance = distance()
    t = threading.Thread(name='getMouseDataThread', target=getMouseData,args=(i,results))
    w = threading.Thread(name='servoControl', target=servoControl,args=(i,test_angles,test_angles2))
 #    for u in range(4):
 #        servo(test_angles)
	# sum1 = 0
	# for t in range(10):
 #        	sum1 += getMotion()
	# sum1 = sum1/10.0
	# acc.append(sum1)
 #        servo(test_angles2)
	# sum2 = 0
 #    	for t in range(10):
 #                sum2 += getMotion()
	# sum2 = sum2/10.0
 #        acc.append(sum2)
        #servo(test_angles3)
    t.start()
    w.start()

    w.join()
    t.join()

    w.exit()
    t.exit()
    # sum_array.append(np.sum(results[i],axis=0))
    # print(sum_array[i])
    # acc_max = max(acc)
    # cur_distance = distance()
    #last_s = test_angles.copy()
    # div_distance = cur_distance - pre_distance
    a.append([results[i][0],results[i][1],[test_angles,test_angles2]])
    # if div_distance > max_dis:
    #     max_dis = div_distance
    #     action1_max = test_angles
    #     action2_max = test_angles2

    # print(acc_max)
    # acc = []
    servo([0,0,0,0])   
    sleep(.5)

    #np.savetxt('newangles_m.txt', angles)
    #np.savetxt('newsenses_m.txt', senses)
#a_sorted = sorted(a, key=itemgetter(0),reverse=True)

#with open('outfileMouseBig_results_array_diff', 'wb') as fp:
 # pickle.dump(results, fp)
with open('outfileMouseBig_a_array_diff_with_R_L', 'wb') as fp:
  pickle.dump(a, fp)
#with open('', 'wb') as fp:
#   pickle.dump(sum_array, fp)
#print (sum_array)

#with open ('outfileMouseBig_a_array_diff', 'rb') as fp:
#        a = pickle.load(fp)

#with open ('outfileMouseBig_a_array_diff', 'rb') as fp:
 #       a = pickle.load(fp)
#for c in range(len(a)):
 #	a[c][0] = sum_array1[c][1]
a_sorted = sorted(a, key=itemgetter(0),reverse=True)
 # with open('outfile', 'wb') as fp:
# #     pickle.dump(a, fp)
# with open ('outfile', 'rb') as fp:
#     itemlist = pickle.load(fp)


print  (a_sorted)

for pp in range(0):
     print(pp)
     sleep(5)
     for p in range(40):
 	print(a_sorted[pp])
        #servo(a_sorted[len(a_sorted)-1][1])
        #servo(a[len(a)-1][2])
        #servo(a[len(a)-1][3])
        servo(a_sorted[pp][2][0])
        servo(a_sorted[pp][2][1])

        #servo(a[0][3])

sleep(3)
for pp in range(10):
    print(pp)
    pp=pp+1
    sleep(5)
    for p in range(100):
        print(a_sorted[len(a_sorted)-pp])
        servo(a_sorted[len(a_sorted)-pp][2][0])
        servo(a_sorted[len(a_sorted)-pp][2][1])
        # servo(a[len(a)-1][3])
       #  servo(a[0][1])
       #  servo(a[0][2])
       #  servo(a[0][3])
       # sleep(.1)
#angles = np.loadtxt('newangles.txt')
#senses = np.loadtxt('newsenses.txt')


# a = np.array([ [1, 0, 255, 15, 0, 0] , [1, 0, 255, 31, 0, 0] , [1, 0, 4, 192, 255, 0] , [1, 0, 1, 224, 255, 0] , [1, 0, 4, 240, 255, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 2, 208, 255, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 224, 255, 0] , [1, 0, 254, 15, 0, 0] , [1, 0, 0, 16, 0, 0] , [1, 0, 0, 32, 0, 0] , [1, 0, 3, 240, 255, 0] , [1, 0, 254, 15, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 16, 0, 0] , [1, 0, 253, 31, 0, 0] , [1, 0, 255, 255, 255, 0] , [1, 0, 252, 15, 0, 0] , [1, 0, 252, 31, 0, 0] , [1, 0, 255, 15, 0, 0] , [1, 0, 254, 15, 0, 0] , [1, 0, 255, 15, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 240, 255, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 1, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 4, 0, 0, 0] , [1, 0, 255, 31, 0, 0] , [1, 0, 1, 32, 0, 0] , [1, 0, 5, 208, 255, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 240, 255, 0] , [1, 0, 1, 208, 255, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 251, 175, 0, 0] , [1, 0, 0, 112, 0, 0] , [1, 0, 254, 255, 255, 0] , [1, 0, 0, 240, 255, 0] , [1, 0, 0, 240, 255, 0] , [1, 0, 255, 15, 0, 0] , [1, 4, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 0, 0, 0] , [1, 0, 0, 240, 255, 0] , [1, 0, 254, 31, 255, 0] , [1, 0, 255, 127, 255, 0] , [1, 0, 255, 159, 0, 0] , [1, 0, 3, 64, 0, 0] , [1, 0, 0, 0, 0, 0] ])
# aa = np.array([[10,20,30,40],[20,20,70,80]])
# aaa = [[10,20,30,40],[20,20,70,80]]
# aaa.append([10,20,30,60])
# aa = np.array(aaa)

# b = aa.mean(axis=0)
# print(b)
