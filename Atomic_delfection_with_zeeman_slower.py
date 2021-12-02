import cv2
from PIL import Image
import numpy as np
from scipy import *

##############################
#Declarations and Setup
import seaborn as sns
from random import choices
from math import pi,sqrt,floor,fabs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import*
from scipy import constants




#Universal constants
k_B = constants.k # Boltzmann constant
mu_0 = constants.mu_0 #permeability of free space
hbar = constants.hbar #reduced Planck's constant
mu_B = (constants.physical_constants['Bohr magneton'])[0] # Bohr magneton in SI units
a_m = (constants.physical_constants['atomic mass constant'])[0] # Atomic mass constant in kg
c = constants.c #speed of light in vacuum

#Global experimental parameters
T = 8 # temperature of oven
beam_radius = 0.5 # radius of atomic beam in cm
m_Sr = 86.9088774970*a_m; # atomic mass in kg
w_c = 460.8622e-9 # cooling wavelength in m
k_wave=2*pi/w_c; # wave number
Gamma = 2*pi*32e6 # decay rate of cooling transition in angular frequency (includes factor of 2*pi)
delta = -58e6; # Detuning
dt = 5e-6 # Time step


#Saturation intensity of cooling transition
Isat = 2 * hbar * (pi**2) * c * Gamma / (3 * (w_c**3)) # in SI
I_laser = 0.1*Isat;
s0 = I_laser/Isat;
B = 0.5 ; # Testla/cm (50 G/cm)
w_beam = 10e-3 # Beam waist of laser


###############################################################################
### Import and rescale figure
### rescale picture
img = Image.open('Zeeman_Slower_no.jpg')
img_scale = Image.open('Zeeman_Slower_scale.jpg')
wsize = int(800) # x-axis pixel
hsize = int(500) # y-axis pixel
img = img.resize((wsize, hsize), Image.ANTIALIAS)
img_scale=img_scale.resize((wsize, hsize), Image.ANTIALIAS)
img.save('resized_Zeeman_Slower_no.jpg')
img_scale.save('resized_Zeeman_Slower_scale.jpg')
img = cv2.imread('resized_Zeeman_Slower_no.jpg',2);
img_scale = cv2.imread('resized_Zeeman_Slower_scale.jpg',2);

# change image to black and white
ret, bw_img = cv2.threshold(img,228,255,cv2.THRESH_BINARY)
ret, bw_img_scale = cv2.threshold(img_scale,228,255,cv2.THRESH_BINARY)
bw_img = bw_img.transpose()


x_0= np.linspace(0,1, wsize);
y_0= hsize*np.linspace(1,1, wsize);
j_search = 0;
for i in range(wsize):
    for j in range(hsize):
        if bw_img[i,j]==0:
            y_0[i]=hsize-j;
#Rescale the figure
xmin = -100;
xmax=800;
int_xmin=137; # x= 0;
int_xmax=704; # x=10;
x=np.linspace(xmin,xmax,int_xmax-int_xmin);
y=y_0[int_xmin:int_xmax]/hsize;
y=y[x>0]
x=x[x>0]

v = x;
f_ZS = y;
plt.plot(v,f_ZS)

# Size of laser
xmin_laser = -30e-3; xmax_laser = 30e-3;
ymin_laser = -w_beam/2; ymax_laser = w_beam/2;
# Boundary of plot
xmin_bound = -50e-3; xmax_bound = 50e-3;
ymin_bound = -80e-3; ymax_bound = 50e-3;

def acceleration_x(x,y,vx,vy): # Acceleration due to MOT beam
        # Gaussian function represents a profile of laser
    return -np.exp(-y**2/(2*w_beam**2))/(2*pi*w_beam)*(hbar*k_wave*s0*Gamma/(2*m_Sr))*((1/(1+s0+4*(delta-k_wave*vx-mu_B*B*x)**2/Gamma**2))-(1/(1+s0+4*(delta+k_wave*vx+mu_B*B*x)**2/Gamma**2)))

def acceleration_y(x,y): # No acceleration along y-axis
    return 0



fig = plt.figure(1)
theta =pi/3; # Angle between atomic beam and laser beam
# Initial position of atomic beam
x_traj = [xmin_bound*np.cos(theta)*8/9,] 
y_traj = [ymin_bound*np.sin(theta)*8/9,]
x_F=np.linspace(xmin_bound,xmax_bound,400);
y_F = np.linspace(ymin_bound,ymax_bound,400);
# Plot laser beam
X_F, Y_F = np.meshgrid(x_F, y_F)
laser_beam = np.exp(-(Y_F**2)/(2*w_beam**2))/(2*pi*w_beam)
plt.contourf(X_F*1e3,Y_F*1e3,laser_beam,50,cmap=cm.Blues)
plt.xlim([xmin_bound*1e3, xmax_bound*1e3])
plt.ylim([ymin_bound*1e3, ymax_bound*1e3])

v_rand = []
v_end =[]

# Calculation of atomic beam trajectory
for i in range(3000):
    # Randomly choose atom under the Zeeman slower distribution 
    v_atom = choices(v,f_ZS)[0]
    v_rand += [v_atom,]
    # Trajectory with laser beam
    vx_traj = [v_atom*np.cos(theta),]
    vy_traj = [v_atom*np.sin(theta),]
    y_traj = [ymin_bound*np.sin(theta)*8/9,];
    x_traj = [y_traj[0]*vx_traj[0]/vy_traj[0],];
    theta =pi/3

    while x_traj[-1] > xmin_bound and x_traj[-1] < xmax_bound and y_traj[-1] > ymin_bound and y_traj[-1] < ymax_bound :
        vx_update = vx_traj[-1] - dt * acceleration_x(x_traj[-1],y_traj[-1],vx_traj[-1],vy_traj[-1])
        vy_update = vy_traj[-1] - dt * acceleration_y(x_traj[-1],y_traj[-1])
        x_update = x_traj[-1] + dt * vx_traj[-1]
        y_update = y_traj[-1] + dt * vy_traj[-1]
        x_traj += [x_update,]
        y_traj += [y_update,]
        vx_traj += [vx_update,]
        vy_traj += [vy_update,]
    if vx_traj[-1] <0.1:
        # The criteria of deflected atomic beam is velocity in x-axis very small 
        # In this case I count every atoms which have velocity in x-axis<0.1 m/s
        v_end+=[sqrt(vx_traj[-1]**2+vy_traj[-1]**2),]
    plt.plot(np.array(x_traj)*1e3,np.array(y_traj)*1e3, label='%s m/s' % v_atom)
plt.savefig('atomic_deflection.jpg', dpi=300)
plt.show()


v_rand =np.array(v_rand);

# Plot velocity distribution of Zeeman slower
fig = plt.figure(2)
sns.histplot(v_rand, stat='density',bins =30)
plt.plot(v,f_ZS/np.trapz(f_ZS,v),'r')
plt.ylim(0, 0.002)

# Plot velocity distribution of 2D MOT (Red line is the velocity distribution of Zeeman slower)
fig = plt.figure(3)
sns.histplot(v_end, stat='density',bins =50)
plt.plot(v,f_ZS/np.trapz(f_ZS,v),'r')

