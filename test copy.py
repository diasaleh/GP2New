# import threading
# import time
# import logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-10s) %(message)s',
#                     )
# def daemon():
# 	while True:
		
# 	    logging.debug('Starting')
# 	    time.sleep(.2)
# 	logging.debug('Exiting')

# d = threading.Thread(name='daemon', target=daemon)
# d.setDaemon(True)

# def non_daemon():
#     logging.debug('Starting')
#     time.sleep(2)
#     logging.debug('Exiting')

# t = threading.Thread(name='non-daemon', target=non_daemon)

# d.start()
# t.start()

# d.join()
# t.join()
# import multiprocessing
# import time
# e=0
# def getMouseData(i):
# 	a=[]
# 	while True :
# 		a.append(i+1)
# 		print ("getMouseData "+str(i)+ " Starting")
# 		time.sleep(.1)
# 	return a
# 	print ("getMouseData "+str(i) + " Exiting")

		
# def servo(i):
#     print ("servo "+str(i)+ " Starting")
#     time.sleep(2) 
#     print ("servo "+str(i)+ " Exiting")

# # for i in range(4):
# p = multiprocessing.Process(target=getMouseData, args=(0,))
# p.start()
# servo(0)
# servo(1)
# e=1
# #t = threading.Thread(name='getMouseDataThread', target=getMouseData,args=(i,))
# #w = threading.Thread(name='servo', target=servo,args=(i,))
# # t.setDaemon(True)
# p.terminate()
# p.join()
import threading
import time
learning_episodes = 5
exitFlag = [0]*learning_episodes
results = [None] * learning_episodes
def getMouseData(idd,results):
	a=[]
	while not exitFlag[idd]:
	    a.append([idd,idd,idd,idd,idd])
	    print (threading.currentThread().getName(), 'Starting '+str(idd))
	    time.sleep(.5)
	results[idd] = a
	print (threading.currentThread().getName(), 'Exiting '+str(idd))

		
def servo(idd):
    print (threading.currentThread().getName(), 'Starting '+str(idd))
    time.sleep(2)
    print (threading.currentThread().getName(), 'Exiting '+str(idd))
    exitFlag[idd] = 1

for i in range(learning_episodes):
	print(i)
	t = threading.Thread(name='getMouseDataThread', target=getMouseData,args=(i,results))
	w = threading.Thread(name='servo', target=servo,args=(i,))
	# t.setDaemon(True)
	w.start()
	t.start()
	w.join()
	t.join()
print (results)

# #!/usr/bin/python

# import sys
# is_py2 = sys.version[0] == '2'
# if is_py2:
#     import queue as queue
# else:
#     import queue as queue
# import threading
# import time

# exitFlag = 0

# class myThread (threading.Thread):
#     def __init__(self, threadID, name, q):
#        threading.Thread.__init__(self)
#        self.threadID = threadID
#        self.name = name
#        self.q = q
#     def run(self):
#        print ("Starting " + self.name)
#        process_data(self.name, self.q)
#        print ("Exiting " + self.name)
# def process_data(threadName, q):
#     while not exitFlag:
#         queueLock.acquire()
#         if not workqueue.empty():
#            data = q.get()
#            queueLock.release()
#            print ("%s processing %s" % (threadName, data))
#         else:
#            queueLock.release()
#         time.sleep(1)

# threadList = ["Thread-1", "Thread-2", "Thread-3"]
# nameList = ["One", "Two", "Three", "Four", "Five"]
# queueLock = threading.Lock()
# workqueue = queue.queue(10)
# threads = []
# threadID = 1

# # Create new threads
# for tName in threadList:
#    thread = myThread(threadID, tName, workqueue)
#    thread.start()
#    threads.append(thread)
#    threadID += 1

# # Fill the queue
# queueLock.acquire()
# for word in nameList:
#    workqueue.put(word)
# queueLock.release()

# # Wait for queue to empty
# while not workqueue.empty():
#    pass

# # Notify threads it's time to exit
# exitFlag = 1

# # Wait for all threads to complete
# for t in threads:
#    t.join()
# print ("Exiting Main Thread")
# from threading import Thread

# def func(argument):
#     while True:
#         print(argument)

# def main():
#     Thread(target=func,args=("1",)).start()
#     Thread(target=func,args=("2",)).start()
#     Thread(target=func,args=("3",)).start()
#     Thread(target=func,args=("4",)).start()

# if __name__ == '__main__':
#     main()
# #!/usr/bin/python

# import threading
# import time

# class myThread (threading.Thread):
#    def __init__(self, threadID, name, counter):
#       threading.Thread.__init__(self)
#       self.threadID = threadID
#       self.name = name
#       self.counter = counter
#    def run(self):
#       print ("Starting " + self.name)
#       # Get lock to synchronize threads
#       threadLock.acquire()
#       print_time(self.name, self.counter, 3)
#       # Free lock to release next thread
#       threadLock.release()

# def print_time(threadName, delay, counter):
#    while counter:
#       time.sleep(delay)
#       print ("%s: %s" % (threadName, time.ctime(time.time())))
#       counter -= 1

# threadLock = threading.Lock()
# threads = []

# # Create new threads
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)

# # Start new Threads
# thread1.start()
# thread2.start()

# # Add threads to thread list
# threads.append(thread1)
# threads.append(thread2)

# # Wait for all threads to complete
# for t in threads:
#     t.join()
# print ("Exiting Main Thread")
# import threading
# import time
# def worker(num):
#     """thread worker function"""
#     time.sleep(1)
#     print ('Worker: %s' % num)

#     return
# def worker2(num):
#     """thread worker function"""
#     print ('Worker2: %s' % num)
#     return

# threads = []
# t = threading.Thread(target=worker2, args=(19,))
# t.start()
# for i in range(5):

#     t = threading.Thread(target=worker, args=(i,))
#     threads.append(t)
#     t.start()
    # worker(i)
# import array
# import sys
# import usb.core
# import usb.util

# VID = 0x0
# PID = 0x538
# DATA_SIZE = 4

# # printina modulio vidurius :for i in dir(usb.util): print i

# # try to find Logiceh USB mouse
# device = usb.core.find(idVendor = VID, idProduct = PID)
# if device is None:
#     sys.exit("Could not find Logitech USB mouse.")

# # make sure the hiddev kernel driver is not active
# if device.is_kernel_driver_active(0):
#     try:
#         device.detach_kernel_driver(0)
#     except usb.core.USBError as e:
#         sys.exit("Could not detatch kernel driver: %s" % str(e))

# # set configuration
# try:
#     device.reset()
#     device.set_configuration()
# except usb.core.USBError as e:
#     sys.exit("Could not set configuration: %s" % str(e))

# endpoint = device[0][(0,0)][0]
# print (endpoint.wMaxPacketSize)


# data = array.array('B',(0,)*4)
# while data[0] != 3:
#     try:
#         data = device.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize)
#         print (data)
    
#     except usb.core.USBError as e:
#         if e.args == ('Operation timed out',):
#             print ("timeoutas")
#             continue
# print ("baigiau! :)")
# from pynput.mouse import Listener

# def on_move(x, y):
#     print('Pointer moved to {0}'.format(
#         (x, y)))

# def on_click(x, y, button, pressed):
#     print('{0} at {1}'.format(
#         'Pressed' if pressed else 'Released',
#         (x, y)))
#     if not pressed:
#         # Stop listener
#         return False

# def on_scroll(x, y, dx, dy):
#     print('Scrolled {0}'.format(
#         (x, y)))

# # Collect events until released
# with Listener(
#         on_move=on_move,
#         on_click=on_click,
#         on_scroll=on_scroll) as listener:
#     listener.join()



'''
import operator
import collections
from operator import itemgetter
import numpy as np
import random
import pickle

# d={}
# for i in range(10):
# 	d[i] = [i+4,[i,i,i,i,i]]
# d[100] = "haha"
# print (d)
# od = collections.OrderedDict(sorted(d.items(), reverse=True))
# print("\n\n")
# print (od)
# a=[]
# a = [[13, [2,1,4324,2]], [332, [424,423,4324]], [17, [1,1,1,1]]]
# print (a)
# a.append( [131,[31,31,31]])
# print(a[0])
# a = sorted(a, key=itemgetter(0),reverse=True)
# print(a)
# print (a[0][1][0])
a=[]
def distance():
	return random.random() 
def servo(x):
	print (x)
for z in range(100):
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

    div_distance = cur_distance - pre_distance
    a.append([div_distance,test_angles,test_angles2])
thefile = open('test.txt', 'w')
print("\n\n\n\n\n")
a = sorted(a, key=itemgetter(0),reverse=True)
with open('outfile', 'wb') as fp:
    pickle.dump(a, fp)
print("fdsadfsdafdsfdsa\n\n")
with open ('outfile', 'rb') as fp:
    itemlist = pickle.load(fp)
print(itemlist[0][1])
# for pp in range(5):
    # for p in range(50):
    #     servo(a[pp][1])
    #     servo(a[pp][2])
    #     # sleep(.2)

 '''