import time
from mpu6050 import mpu6050
sensor = mpu6050(0x68)
mpu = mpu6050(0x68)

acceleration = []
velocity = [0]
timee = 0.2

def sleep(x):
    time.sleep(x)

for i in range(1000):
 #	   print(mpu.get_temp())
    accel_data = mpu.get_accel_data()
   # print(accel_data['x'])
   # print(accel_data['y'])
   # print(accel_data['z'])
    gyro_data = mpu.get_gyro_data()
    print(gyro_data['y'])
    #print(gyro_data['y'])
    #print(gyro_data['z'])
    sleep(timee)


