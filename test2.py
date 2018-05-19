import numpy as np
comp = np.c_[tuple(i.ravel() for i in np.mgrid[:2,:2,:2,:2,:2,:2,:2,:2])]
for i in range(256):
        print("\n\n"+str(i)+"\n\n")
        test_angles = np.array(comp[i])
        #test_angles = np.zeros(servo_num)
        test_angles2 = np.zeros(8)
        #test_angles = np.random.choice([-.15,.15],servo_num)
   #     test_angles2 = np.random.choice([-.25,0,.25],servo_num)

        for v in range(8):
        	test_angles2[v]=not (test_angles[v])

        
        for v in range(8):
            if test_angles2[v] == 0:
                test_angles2[v] = .15
            if test_angles2[v] == 1:
                test_angles2[v] = -.15
            if test_angles[v] == 0:
                test_angles[v] = .15
            if test_angles[v] == 1:
                test_angles[v] = -.15
        print (test_angles )
        print (test_angles2)