    #Loading matplotlib
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

#Loading other useful libraries

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
from pims import ImageSequence
import trackpy as tp

from scipy import ndimage
from scipy import stats as sts
from scipy.optimize import curve_fit

from skimage import morphology
from skimage import io

from PIL import Image
import math



#@pims.pipeline


    


    

#frames=pims.ImageSequence(r'\\elite.lakeforest.edu\facstaffhome$\PhysicsUser\Marcin Summer 2021\videos\00010\*.tif')
frames=pims.ImageSequence('videos/00010/*.tif')
#image=frames[159] 
frame_no=0
store_green=[]
store_red=[]
store_white=[]

x_init=27
x_fin=113
#x_fin=x_fin+1

#each frame is in a different layer 
#each particle has a different row 
#each column represents different data on that particle at that time.
#the columns are position x comp, position y comp, velocity x comp, velocity y comp
data_w=np.zeros((148,250,4))

for i in range(148):
#for image in frames:
    
        image=frames[i]
        frame_no=frame_no+1
        green_image=image[:,:,1]
        #plt.imshow(green_image)
        #plt.savefig('green_image')
        red_image=image[:,:,0]
        #plt.imshow(red_image)
        #plt.savefig('red_image')
        green_minus_red=green_image-red_image
        #plt.imshow(green_minus_red)
        #plt.savefig('green_minus_red')
        red_minus_green=red_image-green_image
        #plt.imshow(red_minus_green)
        #plt.savefig('red_minus_green')
        #plt.imshow(image)
        
        
        #WHITE DOTS
        blue_image=image[:,:,2]
        white=blue_image>250   
        #plt.imshow(blue_image)
        #plt.savefig('blue_image')
        
        #GREEN DOTS
        grn_greater_than_red=green_image
        np.greater(green_image,red_image,out=grn_greater_than_red)
        green_minus_red_img = green_minus_red * grn_greater_than_red
        #plt.imshow(green_minus_red_img, cmap=plt.cm.gray)
        #plt.savefig('green_minus_red_img')
            
        #RED DOTS
        red_greater_than_green=red_image
        np.greater(red_image,green_image,out=red_greater_than_green)
        red_minus_green_wo_red=red_minus_green*grn_greater_than_red
        red_minus_green_with_red=red_minus_green*red_greater_than_green
        #plt.imshow(red_minus_green_with_red)
        #plt.savefig('red_minus_green_with_red')
        #plt.imshow(red_minus_green_wo_red)
        #plt.savefig('red_minus_green_wo_red')
        red_minus_green_img=red_minus_green_with_red-red_minus_green_wo_red
        #plt.imshow(red_minus_green_img)
        #plt.savefig('red_minus_green_img')
        
        
        green_blur=ndimage.gaussian_filter(green_minus_red_img, sigma=5) #makes dots more round
        green_bin=green_blur>50 #creates binary image
        green_open=ndimage.binary_opening(green_bin) #gets rid of small white pixels all around
        green_close=ndimage.binary_closing(green_open) #gets rid of small black spots within white dots
        #plt.imshow(green_close)
        '''
        plt.imshow(green_blur)
        plt.savefig('green_blur')
        plt.imshow(green_bin)
        plt.savefig('green_bin')
        plt.imshow(green_open)
        plt.savefig('green_open')
        plt.imshow(green_close)
        plt.savefig('green_close')
        '''    
        red_blur=ndimage.gaussian_filter(red_minus_green_img, sigma=5)
        red_bin=red_blur>50
        red_open=ndimage.binary_opening(red_bin)
        red_close=ndimage.binary_closing(red_open)
        #plt.imshow(red_close)
        '''
        plt.imshow(red_blur)
        plt.savefig('red_blur')
        plt.imshow(red_bin)
        plt.savefig('red_bin')
        plt.imshow(red_open)
        plt.savefig('red_open')
        plt.imshow(red_close)
        plt.savefig('red_close')
        '''    
        white_blur=ndimage.gaussian_filter(blue_image, sigma=3)
        white_bin=white_blur>200
        white_open=ndimage.binary_opening(white_bin)
        white_close=ndimage.binary_closing(white_open)
        #plt.imshow(white_close)
        #plt.savefig('white_close')
        #Turn black regions outside of the particle arena
        white_close[:50]=0
        white_close[:,:120]=0
        white_close[580:720]=0
        white_close[:,1200:1280]=0
        '''
        plt.imshow(white_blur)
        plt.savefig('white_blur')
        plt.imshow(white_bin)
        plt.savefig('white_bin')
        plt.imshow(white_open)
        plt.savefig('white_open')
        plt.imshow(white_close)
        plt.savefig('white_close')
        '''
        #return white_close
    #tp.batch(preprocess, 15)
    
        #plt.imshow(white_close)
        
        #Now, the graphic manipulation of frames is done. Next, the code finds the dots.
        
        red_loc=tp.locate(red_close, 15, invert=False)
        red_marked=tp.annotate(red_loc, red_close, color='red', plot_style={'markersize':8})
        
        #finding the angle of rotation (goal: red axis || horizontal)
        yi=red_loc.at[0,'y']
        yf=red_loc.at[15,'y']
        xi=red_loc.at[0,'x']
        xf=red_loc.at[15,'x']
        d=math.sqrt((xf-xi)**2+(yf-yi)**2)
        theta=math.acos((xf-xi)/d)
        #print(theta)
        
        y_red=red_loc[['y']].to_numpy() #getting coordinates 
        x_red=red_loc[['x']].to_numpy()
        y_prime_red=y_red*math.cos(theta)-x_red*math.sin(theta)#rotating coordinates
        x_prime_red=x_red*math.cos(theta)+y_red*math.sin(theta)
        red_loc["y'"]=y_prime_red #passing coordinates to red_loc
        red_loc["x'"]=x_prime_red
        red_loc['frame']=int(frame_no) #attaching number of frame
        red_loc_tot=store_red.append(red_loc)
        
        green_loc = tp.locate(green_close, 31, invert=False)
        green_marked=tp.annotate(green_loc, green_close, color='green', plot_style={'markersize':8})
    
        y_green=green_loc[['y']].to_numpy()
        x_green=green_loc[['x']].to_numpy()
        y_prime_green=y_green*math.cos(theta)-x_green*math.sin(theta)
        x_prime_green=x_green*math.cos(theta)+y_green*math.sin(theta)
        green_loc["y'"]=y_prime_green
        green_loc["x'"]=x_prime_green
        green_loc['frame']=int(frame_no)
        green_loc_tot=store_green.append(green_loc)
        
        white_loc=tp.locate(white_close, 15, invert=False)
        white_marked=tp.annotate(white_loc, white_close, color='purple', plot_style={'markersize':8})
        
        y_white=white_loc[['y']].to_numpy()
        x_white=white_loc[['x']].to_numpy()
        y_prime_white=y_white*math.cos(theta)-x_white*math.sin(theta)
        x_prime_white=x_white*math.cos(theta)+y_white*math.sin(theta)
        white_loc["y'"]=y_prime_white
        white_loc["x'"]=x_prime_white
        white_loc['frame']=int(frame_no)
        white_loc_tot=store_white.append(white_loc)
  

'''  

#Checking mechanism. If you want to produce an image of all marked dots:
#-uncomment this paragraph
#-pick one frame to analyze (line 37)
#-disable the for loop
#tp.annotate produces three images of marked dots. In the interactive top-right Spyder window, you need to 'save all plots as...' to a separate folder with check-images. Then copy the front name (omit the parentheses and number) of the created files into line 156.
#all_dots produces an image of all identified and marked dots. Use it to compare with original frame 'image'(line 52)
check=pims.open(r'\\elite.lakeforest.edu\facstaffhome$\PhysicsUser\Marcin Summer 2021\videos\check\Figure 2021-06-11 143459 (*).png')
green_check=check[0]
red_check=check[1]
white_check=check[2]
all_dots=green_check+red_check+white_check
plt.imshow(all_dots)
plt.axis("off")

'''

#Creating 3 dataframes with all locations, all particles in all frames 
store_green=pd.concat(store_green, ignore_index=False)
store_red=pd.concat(store_red, ignore_index=False)
store_white=pd.concat(store_white, ignore_index=False)

search_range=30
green_trajectory=tp.link(store_green, search_range, pos_columns=["y'","x'"]);
red_trajectory=tp.link(store_red,search_range, pos_columns=["y'","x'"]);
#pred = tp.predict.ChannelPredict(30, 'x', minsamples=3)
white_trajectory=tp.link(store_white,15, pos_columns=["y'","x'"], memory=0,adaptive_stop=2, adaptive_step=0.99);

#Next section is devoted to finding speed of the pusher.

#Plotting position of green-rake-dots vs. time
top_right_green=green_trajectory[green_trajectory['particle']==1]
top_right_green.plot(kind='scatter',x='frame', y="x'", xlim=(x_init,x_fin),grid=True)
y_arr1=top_right_green["x'"].to_numpy()
x_arr1=top_right_green['frame'].to_numpy()
y_arr1=y_arr1[x_init:x_fin] #exlude regions where dot is stationary
x_arr1=x_arr1[x_init:x_fin] 
slope1, intercept, r, p, se=sts.linregress(x_arr1,y_arr1)
#print(slope1)

bottom_right_green=green_trajectory[green_trajectory['particle']==3]
bottom_right_green.plot(kind='scatter',x='frame', y="x'", xlim=(x_init,x_fin),grid=True)
y_arr3=bottom_right_green["x'"].to_numpy()
x_arr3=bottom_right_green['frame'].to_numpy()
y_arr3=y_arr3[x_init:x_fin] #exlude regions where dot is stationary
x_arr3=x_arr3[x_init:x_fin] 
slope3, intercept, r, p, se=sts.linregress(x_arr3,y_arr3)
#print(slope3)

#Scaling distance on the image to real separation of red dots
y_array=red_loc["x'"].to_numpy()
y_array=y_array[0:16] #selecting only top row dots
x_array=[0,1,2,3,4,5,6,8,10,12,13,14,15,16,17,18]
#plt.plot(x_array,y_array, 'ro')
#plt.xticks(np.arange(min(x_array), max(x_array)+1, 1.0))
#plt.grid()
#plt.show()
l_conv, intercept1, r1, p1, se1=sts.linregress(x_array,y_array)
l_conv=l_conv/2.54 #[px/cm]
#print("Lenght conversion factor:", l_conv)

vel_gdot1=(abs(slope1)/l_conv)*29*10 #speed of the first green dot [mm/s]
vel_gdot3=(abs(slope3)/l_conv)*29*10 #speed of the second green dot [mm/s]
print("v1=", vel_gdot1)
print("v2=", vel_gdot3)

#The next task is to look at the motion of all individual particles. The code will plot 
#horizontal component of position vs. time and use the slope of this graph to plot horizontal
#component of speed vs. time. Then the speed data is stored in white_trajcetory DataFrame and
# used to plot horizontal component of speed vs. position.



for i in range (1,148):
        for particle in range(250):
            one_part=white_trajectory[(white_trajectory["particle"]==particle)&(white_trajectory["frame"]==i)]
            index=one_part.index.tolist()[0] #gets index of one_part, converts it to a list and takes its first element
            data_w[i-1,particle,0]=one_part.at[index,"y'"]
            data_w[i-1,particle,1]=one_part.at[index,"x'"]

for particle in range(0,250):
        for i in range(0,134):
            print('particle = ',particle)
            print('i = ', i)
            #create a window for slope calculation
            t=np.arange(i,i+8)
            x=data_w[i:i+8, particle,1]
            v_x, intercept, r, p, se=sts.linregress(t,x)
            print(v_x)
            data_w[i, particle, 3]=v_x
            plt.scatter(data_w[i,:,1],data_w[i,:,3])
        
t=np.arange(148) # creating an array of frame numbers

for particle in range(250):
        x_pos=data_w[:,particle,1]
        x_vel=data_w[:,particle,3]
        plt.scatter(t/29,x_pos/l_conv) #plot x componenet of position vs. time
        plt.title(particle) #shows the number of particle observed throughout jamming
        plt.xlabel('time(frame)')
        plt.ylabel('x')
        plt.grid()
        plt.show()
        
        plt.scatter(t/29,x_vel*29/l_conv) #plot x componenet of velocity vs. time
        plt.title(particle) 
        plt.xlabel('time(frame)')
        plt.ylabel('v_x')
        plt.grid()
        plt.show()
        #plt.savefig('Marcin Summer 2021/images/run 2')
    
def logistic_curve(x,a,b,c,d):
        return (a/(1+np.exp(c*(x+d))))+b
front_center=[]  
unc=[]
for i in range(61,90):
        xpad=data_w[i,:,1] #X Position of All Dots (in ith frame)
        vxpad=data_w[i,:,3] #vx position all dots
        #xpad=xpad/l_conv
        #vxpad=vxpad/v_conv
        '''
        plt.scatter(xpad,vxpad)
        plt.title(i)
        plt.xlabel('x')
        plt.ylabel('v_x')
        plt.grid()
        #plt.show()
        #plt.savefig()
        '''
        #binning the data based on x position: bin_mean is the mean values of speed in each bin, bin_center is the x mid-values of each bin
        bin_mean ,bin_edges, bin_number = sts.binned_statistic(xpad,vxpad, statistic="mean", bins=20)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        
        #plt.show()
        
        std_devs=[]
        for j in range(1,20):
            binned=np.where(bin_number==j)
            std_dev=np.std(vxpad[binned])
            std_devs.append(std_dev)
        plt.errorbar(bin_centers,bin_mean, yerr=std_dev,xerr=None,fmt='o',color='orange',ecolor='black')
        p0=[8,-8,1,-500]
        params, cov = curve_fit(logistic_curve, bin_centers, bin_mean, p0=p0)
        unc.append(np.sqrt(cov[3,3]))
        a,b,c,d=params[0], params[1], params[2], params[3]
        uai=logistic_curve(bin_centers,a,b,c,d)
        #plt.scatter(bin_centers,bin_mean)
        
        plt.scatter(xpad,vxpad)
        #plt.scatter(bin_centers, bin_mean,linewidth=5)
        plt.plot(bin_centers,uai,color='r')
        plt.vlines(-d, -8, 0, 'k', ':')
        plt.title(i)
        plt.xlabel('x')
        plt.ylabel(r'$v_{x}$')
        plt.grid()
        plt.show()
        front_center.append(-d)
framing=np.arange(61,len(front_center)+61)
front_vel,front_int, rf, pf, sef=sts.linregress(framing,front_center) 
plt.plot(framing,front_int+front_vel*framing,'r',label='-155')
plt.ylabel(r'$x_{f}$')
plt.xlabel('frame') 
plt.title('Jamming Front - horizontal componenet of position vs. time')
plt.grid()
plt.errorbar(framing,front_center, yerr=unc,xerr=None,fmt='o',ecolor='black')   
plt.scatter(framing,front_center,label='data')
    


       
    #time.sleep(1)

'''
#vel_all_dots=[]
for particle_no in range(0,250):
    one_dot=white_trajectory[white_trajectory['particle']==particle_no]
    #one_dot.plot(kind='scatter', x='frame', y="x'", xlim=(x_init,x_fin),grid=True)
    one_dot.plot(kind='scatter', x='frame', y="x'",grid=True, title=particle_no)
    x_one_dot=one_dot["x'"].to_numpy()
    frame_one_dot=one_dot['frame'].to_numpy()
    vel_one_dot=[]
    for i in range(0,frame_no-14):
        x_one_dot=one_dot["x'"].to_numpy()
        frame_one_dot=one_dot['frame'].to_numpy()
        start=i
        stop=i+8
        #window=slice(i,i+8)
        x_one_dot=x_one_dot[start:stop]
        frame_one_dot=frame_one_dot[start:stop]
        w_vel, intercept_vel, r_vel, p_vel, se_vel=sts.linregress(frame_one_dot,x_one_dot)
        #print(i,': w_vel=',w_vel)
        vel_one_dot.append(w_vel)
        #vel_all_dots.append(w_vel)
        #vel_all_dots.append(w_vel)
        #frame=white_trajectory['frame']
        #particle_list=white_trajectory['particle'].to_numpy()
        #white_trajectory = white_trajectory[(white_trajectory['particle'].isin(particle_no))]
        
        #while white_trajectory.loc[white_trajectory['frame']==i]:
            #white_trajectory.loc[white_trajectory['particle']==particle_no, 'speed']=w_vel
        #if i==frame:
            #create a new column speed in white_trajectory and assign the value at correct
            #indices
        #someRowNumber=777
        #for n in range(0,someRowNumber):
            #white_trajectory.loc[white_trajectory.index[someRowNumber], 'speed'] = vel_all_dots
            #print(5)
    #print(vel_one_dot)
    #store_vel_all_dots=pd.concat(store_vel_all_dots)
    for j in range(0,14):
        vel_one_dot.append(0)
        #vel_all_dots.append(0)
        
    #vel_one_dot.append([0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],)
    one_dot['speed']=vel_one_dot
    one_dot.plot(kind='scatter', x='frame', y='speed',grid=True, title=particle_no)
    #one_dot_one_frame=one_dot[one_dot['frame']==1]
    #one_dot_one_frame.plot(kind='scatter', x="x'", y='speed',grid=True, title=particle_no)
'''    
'''   
for frame_num in range(1,149):
    one_frame=white_trajectory[white_trajectory['frame']==frame_num]
    one_frame.plot(kind='scatter', x="x'", y='speed', grid=True, title=frame_num)
    #one_dot.plot(kind='scatter', x='frame', y="x'", xlim=(x_init,x_fin),grid=True)
    #one_dot.plot(kind='scatter', x='frame', y="x'",grid=True, title=particle_no)
    #x_one_dot=one_dot["x'"].to_numpy()
    #frame_one_dot=one_dot['frame'].to_numpy()
    #vel_one_dot=[]
    
    #for i in range(0,250):
        #one_frame.plot(kind='scatter', x="x'", y='speed', grid=True, title=frame_num)
        '''    
'''    
        x_one_dot=one_dot["x'"].to_numpy()
        frame_one_dot=one_dot['frame'].to_numpy()
        start=i
        stop=i+8
        #window=slice(i,i+8)
        x_one_dot=x_one_dot[start:stop]
        frame_one_dot=frame_one_dot[start:stop]
        w_vel, intercept_vel, r_vel, p_vel, se_vel=sts.linregress(frame_one_dot,x_one_dot)
        #print(i,': w_vel=',w_vel)
        vel_one_dot.append(w_vel)
    #print(vel_one_dot)
    for j in range(0,14):
        vel_one_dot.append(0)
        
        for j in range(0,14):
            vel_one_dot.append(0)
        #vel_one_dot.append([0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],)
        one_dot['speed']=vel_one_dot
        one_dot.plot(kind='scatter', x='frame', y='speed',grid=True, title=particle_no)
       ''' 
'''
for particle in white_trajectory:
    y_white=white_trajectory.
  
    particle.plot(kind='scatter',x='frame', y="x'", xlim=(x_init,x_fin),grid=True)
    y_arr1=top_right_green["x'"].to_numpy()
    x_arr1=top_right_green['frame'].to_numpy()
    y_arr1=y_arr1[x_init:x_fin] #exlude regions where dot is stationary
    x_arr1=x_arr1[x_init:x_fin] 
    slope1, intercept, r, p, se=sts.linregress(x_arr1,y_arr1)
    #print(slope1)
 '''
 