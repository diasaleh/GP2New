#!/usr/bin/env python
########################################################################                                                                  
# This example is for controlling the GoPiGo robot from a mouse scroll                          
# http://www.dexterindustries.com/GoPiGo/                                                                
# History
# ------------------------------------------------
# Author     Date      		Comments
# Karan      20 Aug 14 		Initial Authoring
# 			                                                         
'''
## License
 GoPiGo for the Raspberry Pi: an open source robotics platform for the Raspberry Pi.
 Copyright (C) 2017  Dexter Industries
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/gpl-3.0.txt>.
'''         
#
########################################################################
import struct
import sys
from gopigo import *
import threading

#Open the stream of data coming from the mouse
file = open( "/dev/input/mice", "rb" );
speed=150

debug = 0	#Print raw values when debugging

#Parse through the fata coming from mouse
#Returns: 	[left button pressed,
#		middle button pressed,
#		right button pressed,
#		change of position in x-axis,
#		change of position in y-axis]
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
#stop()
def getMouseData(idd,results):

	a=[0]*4
	print (threading.currentThread().getName(), 'Starting '+str(idd))
	while( 1 ):
        	[l,m,r,x,y]=getMouseEvent() #Get the inputs from the mouse
        # if debug:
        #     print l,m,r,x,y
        # print x,"\t",y

        #If there is a signinficant mouse movement Up (positive y-axis)
        	if y >20:
            		print("fwd()")  #Move forward
            		a[0] = a[0]+1
        #If there is a signinficant mouse movement Down (negative y-axis)
        	elif y<-20:
            		print("bwd()")  #Move Back
            		a[1] = a[1]+1
        #If there is a signinficant mouse movement Left (positive x-axis)
        	elif x<-20:
            		print("left()") #Move left
           	        a[2] = a[2]+1
        #If there is a signinficant mouse movement Right (negative x-axis)
        	elif x>20:
            		print("right()")    #Move Right
            		a[3] = a[3]+1
        	time.sleep(.01)
		results[idd] = a
	print (threading.currentThread().getName(), 'Exiting '+str(idd))

t = threading.Thread(name='getMouseDataThread', target=getMouseData,args=(0,results))
t.start()
t.join()
