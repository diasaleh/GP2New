import random
import numpy as np
from collections import Counter
import RPi.GPIO as GPIO
import time
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

LR = 1e-3
goal_steps = 30
score_requirement = 0
initial_games = 10

def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 1
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        # ///pre distance
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = [0]*4
            action[0] = random.randrange(0,2)
            
            action[1] = random.randrange(0,2)
            
            action[2] = random.randrange(0,2)
            # SetAngle(action[2])
            action[3] = random.randrange(0,2)
            drive(action)
            # SetAngle(action[3])
            observation = action
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
        # score = //(distance - prev)/70

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here. 
        # all we're doing is reinforcing the score, we're not trying 
        # to influence the machine in any way as to HOW that score is 
        # reached.
        output = [0]*8
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                print(data)
                # convert to one-hot (this is the output layer for our neural network)
                if data[1][0] == 1:
                    output[0] = 0
                    output[1] = 1
                elif data[1][0] == 0:
                    output[0] = 1
                    output[1] = 0

                if data[1][1] == 1:
                    output[2] = 0
                    output[3] = 1
                elif data[1][1] == 0:
                    output[2] = 1
                    output[3] = 0

                if data[1][2] == 1:
                    output[4] = 0
                    output[5] = 1
                elif data[1][2] == 0:
                    output[4] = 1
                    output[5] = 0

                if data[1][3] == 1:
                    output[6] = 0
                    output[7] = 1
                elif data[1][3] == 0:
                    output[6] = 1
                    output[7] = 0
                    
                # saving our training data
                training_data.append([data[0], output])
        scores.append(score)
    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

# def neural_network_model(input_size):

#     network = input_data(shape=[None, input_size, 1], name='input')

#     network = fully_connected(network, 128, activation='relu')
#     network = dropout(network, 0.8)

#     network = fully_connected(network, 256, activation='relu')
#     network = dropout(network, 0.8)

#     network = fully_connected(network, 512, activation='relu')
#     network = dropout(network, 0.8)

#     network = fully_connected(network, 256, activation='relu')
#     network = dropout(network, 0.8)

#     network = fully_connected(network, 128, activation='relu')
#     network = dropout(network, 0.8)

#     network = fully_connected(network, 2, activation='softmax')
#     network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
#     model = tflearn.DNN(network, tensorboard_dir='log')

#     return model


# def train_model(training_data, model=False):

#     X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
#     y = [i[1] for i in training_data]

#     if not model:
#         model = neural_network_model(input_size = len(X[0]))
    
#     model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
#     return model

training_data = initial_population()

# model = train_model(training_data)

# scores = []
# choices = []
# for each_game in range(10):
#     score = 0
#     game_memory = []
#     prev_obs = []
#     for _ in range(goal_steps):
#         if len(prev_obs)==0:
#             action = random.randrange(0,2)
#         else:
#             action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

#         choices.append(action)
                
#         new_observation, reward, done, info = env.step(action)
#         prev_obs = new_observation
#         game_memory.append([new_observation, action])
#         score+=reward
#         if done: break

#     scores.append(score)

# print('Average Score:',sum(scores)/len(scores))
# print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
# print(score_requirement)



# Done.  Terminate all signals and relax the motor.
pwm.stop()

# We have shut all our stuff down but we should do a complete
# close on all GPIO stuff.  There's only one copy of real hardware.
# We need to be polite and put it back the way we found it.
GPIO.cleanup()
