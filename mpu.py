import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D
from keras import backend as K

def servo(x):
    i = 1
    if i == 0:
        i+=1

def sleep(x):
    i = 1
    if i == 0:
        i+=1

def getMotion():
    return np.array([-1,0,1,0,1,0])

def clear_output(wait):
    i = 1
    if i == 0:
        i+=1

    
angles = []
senses = []
s = np.array([-1,0,1,0,1,0,-1,0], dtype=np.float64)
last_s = s.copy()
servo(s)
sleep(1)

for z in range(100):
    rd = np.random.randn(8) * 0.3
    rd[[1,3,5,7]] = 0
    test_angles = np.clip(s+rd, -1,1)

    for i in np.linspace(0,1,10):
        servo((1-i) * last_s + i * test_angles)

    last_s = test_angles.copy()
    sleep(.5)

    sense_sum = np.zeros(3)
    for i  in range(10):
        sense_sum += np.array(getMotion())[:3]
        sleep(.05)

    sense = sense_sum/10.0

    angles.append(test_angles.copy())
    senses.append(sense)

    clear_output(wait=True)
    a = np.array(senses)
    fig = plt.plot(a[:,0], c = 'r')
    fig = plt.plot(a[:,1], c = 'b')
    #plt.show()

axis = 0
u = np.mean(senses, axis = 0)
std = np.std(senses, axis = 0)

senses_norm = (senses - u)/std

single_axis = np.array(senses_norm)[:,axis]
fig = plt.plot(single_axis)

def line_fit(single_axis, joint_states):
    m, b = np.polyfit(single_axis, joint_states, 1)
    
    T = np.linspace(np.min(single_axis), np.max(single_axis))

    fig = plt.scatter(single_axis, joint_states)
    fig = plt.plot(T, [m*t + b for t in T])

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
model.add(Dense(64, input_dim=2)) #two for xy
model.add(Activation('tanh'))
model.add(Dense(8))
model.add(Activation('tanh'))

#for a mean squared error regression problem
model.compile(optimizer='rmsprop',loss='mse')

#model training

axis_vis = senses_norm[:,0]

T = np.linspace(np.min(axis_vis), np.max(axis_vis), 30)

for i in range(20):
    model.fit(senses_norm[:,:2], np.array(angles), verbose=False, epochs=10)

    joint_angles = []
    for m in T:
        joint_angles.append(model.predict(np.array([[m, 0]]))[0])

    clear_output(wait=True)
    fig = plt.plot(T, joint_angles)
    #plt.show()

def tilt(x,y):
    joint_angles = model.predict(mp.array([[x, y]]))[0]
    servo(joint_angles)
    sleep(.1)

T = np.linspace(0, np.pi*2.0 , 20)

for t in T:
    joint_angles = model.predict(np.array([[np.sin(t)*2.5, np.cos(t)*2.5]]))[0]
    servo(joint_angles)
    sleep(.1)


joint_angles = model.predict(np.array([[0, 0]]))[0]
servo(joint_angles)



