#!/usr/bin/env python
# coding: utf-8

# # Spring-Mass Oscillator: Numerical Solutions
# 
# This Jupyter notebook provides an in depth examination of the dynamics of a damped harmonic oscillating system, by implementing and comparing four different numerical methods: Euler, Improved Euler, Verlet, and Euler-Cromer. Each method's efficacy in predicting the system's behavior is assessed through a series of simulations, where we analyse the goodness of fit, similarity to the analytical solution, and amount of energy dispersion. 
# 

# ### The task:
# 
# Write a program to calculate the displacements, for each of the four proposed numerical methods.
# 
# We assume some initial conditions and plot results for a number of calculation steps in order to clearly show the differences between the four methods.
# 
# We investigate the effect of the size of time step h, on the accuracy of different methods.

# ### Physics of the Damped Harmonic Oscillator
# 
# The damped harmonic oscillator is a fundamental model in physics, describing systems where the motion is opposed by a force proportional to the velocity. This includes many real-world applications such as car suspensions, molecular vibrations, and electric circuits. The equation governing this motion is a second-order linear differential equation that models the balance between restoring forces, damping forces, and takes into account any external forces if they are present.
# 
# ### Initial Conditions and Constants
# 
# The simulation starts with the mass on a spring initially stretched to a displacement of zero meters and then released with an initial velocity of -1 m/s. The damping coefficient `b` introduces an exponential decay factor, representing the energy loss in the system due to dissipative forces, namely friction.
# 
# ### Numerical Methods Overview
# 
# Numerical methods are essential for solving ordinary differential equations that do not have a straightforward analytical solution. The four methods allow for us to approximate the continuous equations of motion with discrete steps, and each has its advantages and trade-offs in terms of accuracy, stability, and computational efficiency.
# 
# 

# We start by defining the unique paramaters given, along with appropriate values for constants. In addittion to this we import any neccessary libraries. 

# In[1]:


# Initialisation
import string
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit


# In[2]:


# Defining the global constants in upper case

SPRING_CONSTANT = 0.93
MASS = 5.1
INITIAL_DISPLACEMENT = 0
INITIAL_VELOCITY = -1
INTEGRATION_TIME = 140
TIME_STEPS = np.array([0.005, 0.01, 0.05, 0.1])
DAMPING_CONSTANT = 0.08


# 
# ## Modelling the Physical System
# 
# We start by defining our physical constants and initial conditions. Our system is a mass-spring-damper model, with the following equation of motion:
# 
# $$
# m\ddot{x} + b\dot{x} + kx = 0
# $$
# 
# where:
# - \( m \) is the mass,
# - \( b \) is the damping coefficient,
# - \( k \) is the spring constant,
# - \( x \) is the displacement.
# 
# The initial conditions are $( x(0) = 0  \text{m} )$ and $( \dot{x}(0) = -1 \, \text{m/s} )$.
# 
# This section derives the analytical solution of the damped harmonic oscillator under the given initial conditions. It makes use of complex numbers to account for the damped oscillation's phase and magnitude. This solution provides a benchmark to compare the accuracy and stability of the numerical methods applied later.
# 

# In[3]:


# The analytic solution when b is not b_cr

def analytic_solution(INITIAL_DISPLACEMENT, INITIAL_VELOCITY, step_value):
    """
    Computes the analytical solution of the damped harmonic oscillator.
    
    Parameters:
    INITIAL_DISPLACEMENT (float): Initial displacement of the oscillator.
    INITIAL_VELOCITY (float): Initial velocity of the oscillator.
    step_value (float): Time step value for the simulation.
    
    Returns:
    numpy.ndarray: An array of complex numbers representing the analytical solution over time.
    """
    time = time_value(INTEGRATION_TIME,step_value)
    # Constants
    A = (DAMPING_CONSTANT/(2*1j*cmath.sqrt(4*SPRING_CONSTANT*MASS-DAMPING_CONSTANT**2))+1/2)*INITIAL_DISPLACEMENT+INITIAL_VELOCITY*MASS/(
        cmath.sqrt(4*SPRING_CONSTANT*MASS-DAMPING_CONSTANT**2)*1j) 
    B = INITIAL_DISPLACEMENT-A
    
    X = np.exp(-DAMPING_CONSTANT*time/(2*MASS))*(A*np.exp(1j*cmath.sqrt(4*SPRING_CONSTANT*MASS-DAMPING_CONSTANT**2)*time/(2*MASS))
                              +B*np.exp(-1j*cmath.sqrt(4*SPRING_CONSTANT*MASS-DAMPING_CONSTANT**2)*time/(2*MASS)))

    return X
    


# ## Time Value Array and Integration Setup
# Before delving into the numerical methods, we initialize our simulation environment. This includes setting up a time array that spans the entire duration of our simulation. By discretizing time into steps (step_size), we can simulate the oscillator's motion over time. The number_of_steps is determined by dividing the total integration time by the chosen step size, ensuring that our simulation covers the intended duration.

# In[4]:


# for integration time T and number of steps h we can define 

# Choose how long we are integrating for (in s)
#integration_time = 221
# Choose the step size (in s)
step_size = TIME_STEPS[0]  # units: s
number_of_steps = int(INTEGRATION_TIME/step_size)


def empty_array(nsteps):
    """
    Creates empty arrays for displacement and velocity.
    
    Parameters:
    nsteps (int): Number of simulation steps.
    
    Returns:
    tuple: A tuple containing two numpy arrays for displacement and velocity, respectively.
    """
    
    # Determine how many steps there are (values of i); we need to use int to ensure we have a whole number
    # of them, otherwise the loops won't work
    
    # Create empty arrays ready for the values of x and v
    
    displacement = np.empty(nsteps)
    velocity = np.empty(nsteps)
    return displacement, velocity

# Combine time array initialization
def get_time_array(integration_time, step):
    """
    Generates an array of time points starting from zero up to the integration time with a specified time step.

    Parameters:
    integration_time (float): The total time of integration.
    step (float): The time step size.

    Returns:
    ndarray: An array of time points.
    """
    return np.arange(0, integration_time, step)


def time_value(INTEGRATION_TIME,step_value):
    """
    Generates an array of time values for the given integration time and step.
    
    Parameters:
    INTEGRATION_TIME (float): Total integration time.
    step_value (float): Step size in time.
    
    Returns:
    numpy.ndarray: Array of time values.
    """
    return np.arange(0,INTEGRATION_TIME,step_value)


# ## Plotting Routine 
# The plot_routine function standardizes the plotting of our simulation results. By configuring common plot parameters such as font size, figure size, labels, and grid, this function ensures consistency across all our graphs. This routine is crucial for comparing the outcomes of different numerical methods visually.
# 
# 

# In[5]:


# We use a plot routine to ensure consistent graphs throughout

def plot_routine(xlabel, ylabel):
    """
    Configures the plot with common parameters like labels and grid.
    
    Parameters:
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    plt.rcParams.update({'font.size': 14}) 
    plt.style.use('default') 
    plt.rcParams['figure.figsize'] = (12,3) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers' 
    plt.grid(True) 
    plt.legend() 
    plt.show() 
    
def fig_routine(ax, xlabel, ylabel):
    """
    Configures the plot with common parameters like labels and grid, but for data where we use a figure rather
    than one plot.
    
    Parameters:
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)


# ## Fit Testing
# The comparison of the different methods to the analytical solution allows for us to plot the goodness of fit test, where we are able to analyse the percentage difference from the solution and thereby visualise this on a graph. The lower the percentage of the goodness of fit test, the more accurate the meethod is that is being tested.

# In[6]:


def fit_test(observed, expected):
    """
    Computes the percentage difference between observed and expected values to test the fit quality.
    
    Parameters:
    observed (numpy.ndarray): Array of observed values from the simulation.
    expected (numpy.ndarray): Array of expected values from the analytical solution.
    
    Returns:
    float: The average percentage difference across all points.
    """
    
    # Define a valiable to store the total percentage diffference
    total_percent_difference = 0
    
    # Iterate through each value in the arrays
    for count, value in enumerate(expected):

        # Set value to zero for values that would otherwise return an invalid value error
        if observed[count] == value:
            percent_difference = 0
            
        # Calculate percentage difference between observed and actual values and add it to total difference
        else:
            percent_difference = abs((observed[count] - value) / value) * 100
        total_percent_difference += percent_difference
        
    # Calculate the average difference
    average_difference = total_percent_difference / len(expected)
    return average_difference

def fit_plot(INTEGRATION_TIME, TIME_STEPS, method, plot, color, label):
    """
    Plots the goodness of fit for a numerical method over a range of step sizes.

    Parameters:
    integration_time (float): The total time of integration.
    steps (numpy.ndarray): Array of step sizes.
    method (function): The numerical method function being evaluated.

    Returns:
    None
    """
    # Define an array to store goodness of fit values
    goodness_of_fit = np.zeros(len(TIME_STEPS))
    
    # Calculate displacement for a specified method over range of step sizes 
    
    for i, step in enumerate(TIME_STEPS):
        total_steps = math.ceil(INTEGRATION_TIME / step)
        time_array = get_time_array(INTEGRATION_TIME, step)
        
        displacement, _ = method(total_steps, step, DAMPING_CONSTANT)
        analytical_displacement = analytic_solution(INITIAL_DISPLACEMENT, INITIAL_VELOCITY, step).real
        goodness_of_fit[i] = fit_test(displacement, analytical_displacement)
        
    
    if plot == 'yes':
        # setting axis labels & figures 
        plt.plot(TIME_STEPS, goodness_of_fit, '.r', label='data')
        plot_routine('Step Size (s)','Goodness of Fit (%)')
    elif plot == 'no':
        # setting axis labels & figures 
        plt.plot(TIME_STEPS, goodness_of_fit, label=label, color=color)


# ## Plotting the Methods
# The following cell sets up the plot for each method against the analytical solution, it allows us to visualise the sinusodial solutions produced for each step size, for each method.  This thereby gives us an outlook for understanding which method is the most accurate, before further analysis. In addittion, it allows for comparison of each method and what they are physically doing. 

# In[7]:


def plot_times(INTEGRATION_TIME, TIME_STEPS, method):
    """
    Plots displacement vs. time for a numerical method and compares it to the analytic solution.

    Parameters:
    integration_time (float): The total time of integration.
    steps (numpy.ndarray): Array of step sizes.
    method (function): The numerical method function being evaluated.

    Returns:
    None
    """
    
    plt.figure(figsize=(10, 6))
    
    for i, step in enumerate(TIME_STEPS):
        time = np.arange(0, INTEGRATION_TIME, step)

        number_of_steps = int(INTEGRATION_TIME/step)
        displacement, velocity = method(number_of_steps, step, DAMPING_CONSTANT)
        
        # plotting data from experimental runs 1-3:
        plt.plot(time, displacement, label=f'{method.__name__} h={step}')
        
        # Call the analytic solution
        analytical_displacement = analytic_solution(INITIAL_DISPLACEMENT, INITIAL_VELOCITY, step).real

    title = str(method)
    
    # setting axis labels & figures 
    plt.plot(time, analytical_displacement, 'k--', label='Analytic Solution')
    
    plt.xlim(0,INTEGRATION_TIME)
    
    plot_routine('Time (s)','Displacement (m)')



# 
# ## Energy Conservation Analysis
# 
# A critical aspect of physical simulations is energy conservation. We'll calculate the total mechanical energy at each timestep and plot it to observe how well each method conserves energy.
# 

# We are also able to plot the energy for each method, we assume that the energy will remain constant and therefore shouldn't change, however the different distributions may not reflect this, so we use an energy calculation to check the value of the energy for each displacement and velocity, this will allow for us to analyse the energy in the system and understand if it is conserved as we expect it to be, or if it dissipates, providing further insight to the accuracy of each system 
# 
# the **Energy calculation** is given by
# $$E_{i} = 1/2 \cdot (k\cdot x_{i}^2) + 1/2 \cdot( m \cdot v_{i}^2/2)$$ 
# 
# and we are able to implement this calculation in a function call

# In[8]:


def Energy_Calculation(displacement, velocity):
    """
    Calculates the mechanical energy of the damped harmonic oscillator at each time step.

    Parameters:
    displacement (ndarray): An array containing the displacement values of the oscillator over time.
    velocity (ndarray): An array containing the velocity values of the oscillator over time.

    Returns:
    ndarray: An array containing the total mechanical energy of the oscillator at each time step.
    """
    Energy = (1/2) * (SPRING_CONSTANT * displacement ** 2) + (1/2) * (MASS * velocity ** 2)
    return Energy


# We can now build a plotting function to display the energy function over a time period

# In[9]:



def plot_energy(INTEGRATION_TIME, TIME_STEPS, method):
    """
    Plots the energy of the damped harmonic oscillator over time for different time steps and compares it across
    different numerical methods.

    Parameters:
    INTEGRATION_TIME (float): The total time over which the simulation is run.
    TIME_STEPS (ndarray): An array of time steps to be used in the simulation.
    method (function): The numerical method function to be used for simulation.

    Returns:
    None
    """
    
    plt.figure(figsize=(10, 6))
    for step in TIME_STEPS:
        nsteps = int(INTEGRATION_TIME / step)
        displacement, velocity = method(nsteps, step, DAMPING_CONSTANT)
        time = np.arange(0, INTEGRATION_TIME, step)
        energy = Energy_Calculation(displacement, velocity)  # Ensure this works with arrays

        plt.plot(time, energy, label=f'h={step}')
    
    plt.ylim(0, 4)
    plt.xlim(0, INTEGRATION_TIME)
    plot_routine('Time (s)','Energy (J)')


# 
# ## Numerical Integration Methods
# 
# We'll approach the problem using time-stepping methods that update the system's state over time. We explore the following methods:
# - Euler's Method: A straightforward first-order method.
# - Improved Euler's Method: An improvement over the basic Euler method that uses an average of gradients.
# - Verlet's Method: A second-order method that's more accurate and conserves energy better over long simulations.
# - Euler-Cromer's Method: A variant of Euler's method that calculates the velocity update first.
# 

# ## Euler's Method 
# 
# Euler's method is the simplest one-step procedure for iterating over the differential equations. Although it's straightforward and intuitive, it can lead to significant errors, especially for stiff equations or when a high level of precision is required.
# 

# the **Euler's Method** is 
# $$x_{i+1} = x_{i} + h \cdot v_{i}$$ 
# $$v_{i+1} = v_{i} + h \cdot a_{i}$$    

# Let's implement the method for a spring-mass system described by the equation of motion  
# $ m a = - k x + b v $  
# for some mass $m$, spring constant $k$ and damping constant $b$. Here, velocity and acceleration are obviously 
# $v = \frac{\textrm{d} x}{\textrm{d} t}$ and $a = \frac{\textrm{d} v}{\textrm{d} t}$, and system is subject to some initial conditions 𝑥(𝑡=0) and 𝑣(𝑡=0).
# 
# Assuming h is a small *time step*, the **Euler's Method** is 
# $$x_{i+1} = x_{i} + h \cdot v_{i}$$ 
# $$v_{i+1} = v_{i} + h \cdot a_{i}$$    
# where  
# $$a_{i} = -(k/m) x_{i} -(b/m) v_{i}$$. Here $i$ labels the number of steps taken (each of size $h$ in time). This allows us to model the system over some specified time from $t =0$ to $t=T$.  
# 
# 

# In[10]:


def euler_method(nsteps, hstep, DAMPING_CONSTANT):
    """
    Simulates the motion of a damped harmonic oscillator using the Euler method.

    Parameters:
    nsteps (int): The total number of time steps for the simulation.
    hstep (float): The time increment for each step.
    DAMPING_CONSTANT (float): The damping coefficient of the oscillator.

    Returns:
    tuple: Two ndarrays representing the displacement and velocity of the oscillator at each time step.
    """
    # Complete the initial setup
    
    displacement, velocity = empty_array(nsteps)
    
    # We put in the initial conditions
    
    displacement[0] = INITIAL_DISPLACEMENT # in m
    velocity[0] = INITIAL_VELOCITY # in m/s

    # Calculate the acceleration at step i and the displacement and velocity at step i+1
    for i in range(nsteps - 1):
        acceleration = - (SPRING_CONSTANT / MASS) * displacement[i] - (DAMPING_CONSTANT / MASS) * velocity[i]
        displacement[i+1] = displacement[i] + velocity[i] * hstep
        velocity[i+1] = velocity[i] + acceleration * hstep
    return displacement, velocity 

displacement, velocity = euler_method(number_of_steps, step_size, DAMPING_CONSTANT)


# We can plot the euler's method against the analystical method to get a visualisation of how much they work together

# In[11]:



plot_times(INTEGRATION_TIME, TIME_STEPS, euler_method)

fit_plot(INTEGRATION_TIME, TIME_STEPS, euler_method, 'yes', __, __)

time = np.arange(0, INTEGRATION_TIME, step_size)
plot_energy(INTEGRATION_TIME, TIME_STEPS, euler_method)


# ## Euler Method Analysis
# 
# The first graph shows the Euler Method Plot, For each time step size (h).  The Euler method plots demonstrate its approximation to the oscillator's motion. Smaller steps generally lead to higher accuracy, but computational cost increases.  In addittion, we can see that it is increddibly inaccurate when the step size is 0.1, causing for the damping to seemingly act in the opposite direction. 
# 
# The second graph shows the Goodness of Fit: This metric evaluates how closely the numerical solution aligns with the analytical solution. For larger step sizes, we can see that the goodness of fit test shows that the Euler method becomes increasingly innacurate, demonstrating how the Euler Method is only accurate for small values of step size (close to 0).
# 
# The third graph shows the Energy Dissipation:  Ideally, for a perfectly elastic system without any damping, the energy should remain constant. However, numerical dissipation is evident in the Euler method. It is a particular problem for larger step sizes, where it the system seems to gain energy, which of course is not physically possible or expected in a sinusodial system undergoing damping . This effect highlights the method's limitations in energy conservation.
# 
# These errors are due to numerical errors inherent in the Euler method. We observe fluctuations and a drift from the expected constant energy value.
# 

# # Improved Euler’s Method
# 
# Following from this we are able to write the Improved Eulers method, the second numerical method that we are testing.
# 
# This method improves upon the basic Euler by correcting the estimation of the slope of the solution. It typically provides better accuracy for a slightly increased computational cost.
# 
# 
# An easy way to improve on Euler’s method is to use one extra term in the Taylor expansion for the derivative:
# 
# $$ x(t + h) = x(t) + hx'(t) + \frac{1}{2}h^2x''(t) + \text{error}, $$
# 
# where $\text{error} \sim O(h^3)$.
# 
# Then,
# 
# \begin{align*}
# x_{i+1} &= x_i + hv_i + \frac{h^2}{2}a_i, \\
# v_{i+1} &= v_i + ha_i, \\
# a_i &= -\frac{k}{m}x_i - \frac{b}{m}v_i.
# \end{align*}

# In[12]:


def improved_euler_method(nsteps, hstep, DAMPING_CONSTANT):
    """
    Simulates the motion of a damped harmonic oscillator using the improved Euler method, also known as Heun's
    method.

    Parameters:
    nsteps (int): The total number of time steps for the simulation.
    hstep (float): The time increment for each step.
    DAMPING_CONSTANT (float): The damping coefficient of the oscillator.

    Returns:
    tuple: Two ndarrays representing the displacement and velocity of the oscillator at each time step.
    """
    # Complete the initial setup
    
    displacement, velocity = empty_array(nsteps)
    
    # We put in the initial conditions
    
    displacement[0] = INITIAL_DISPLACEMENT # in m
    velocity[0] = INITIAL_VELOCITY # in m/s

    # Calculate the acceleration at step i and the displacement and velocity at step i+1
    for i in range(nsteps - 1):
        acceleration = - (SPRING_CONSTANT / MASS) * displacement[i] - (DAMPING_CONSTANT / MASS) * velocity[i]
        displacement[i+1] = displacement[i] + velocity[i] * hstep + (hstep**2/2)*acceleration
        velocity[i+1] = velocity[i] + acceleration * hstep
    return displacement, velocity 

displacement, velocity = improved_euler_method(number_of_steps, step_size, DAMPING_CONSTANT)


# In[13]:


plot_times(INTEGRATION_TIME,TIME_STEPS, improved_euler_method)
fit_plot(INTEGRATION_TIME, TIME_STEPS, improved_euler_method, 'yes', __, __)
plot_energy(INTEGRATION_TIME, TIME_STEPS, improved_euler_method)


# ## Improved Euler Method Analysis
# 
# 
# 
# The graphs here show the energy, fit, and plot of the system. When using the Improved Euler method, we expect it to conserve energy better than the basic Euler method, but some numerical dissipation is still expected.
# 
# The first graph shows the improved Euler Method Plot: This method shows a corrected slope for each step, aiming for greater accuracy over the Euler method. The plots for different h values show a closer adherence to the analytical solution, for all values of h.  It is still however, much closer to the analytical values based when the step size is small, however for larger step sizes, it still follows the general shape. 
# 
# The second graph shows the Goodness of Fit, and the third graph shows theEnergy Dissipation: Similar to the Euler method, these analyses for the Improved Euler Method reveal improved accuracy and reduced energy dissipation, showcasing the benefits of the correction step in this method.  However both follow similar analyses as the Euler Method, only to a lesser degree. 
# 

# # Verlet’s Method
# 
# Using Verlet's we can use the terms
# 
# $$ v_i = \frac{x_{i+1} - x_{i-1}}{2h} + O(h^2), $$
# 
# $$ a_i = \frac{2}{h^2}(x_{i+1}+x_{i-1}-2x_i) + O(h^2). $$
# 
# After re-arrange the second expression to get
# 
# $$ x_{i+1} = 2x_i - x_{i-1} + h^2a_i + O(h^4), $$
# 
# $$ a_i = -\frac{k}{m}x_i - \frac{b}{m}v_i, $$
# 
# $$ v_i = \frac{x_{i+1} - x_{i-1}}{2h}. $$
# 
# due to the fact that this is quartic in \(h\), it is much more accurate
# 
# however we can't solve these separately but we are able to rearange for \(x\), since we only need the displacement, and we can combine equations to give
# 
# $$ x_{i+1} = Ax_i + Bx_{i-1}, $$
# 
# where$$ ( A = 2\frac{2m - kh^2}{D} ), ( B = \frac{bh - 2m}{D} ), ( D = 2m + bh ) $$.
# 
# But notice that \( (i+1) \) term requires \( i \) and \( (i-1) \) terms – not self-starting.
# so therefore we must use a different method for the first step, such as improved eulers method.
# 

# 
# ## Verlet Integration
# 
# Verlet integration is particularly popular in molecular dynamics simulations. It is known for its excellent energy conservation properties, especially when simulating systems over long timescales.
# 
# As we will see later, Verlet's method is the most accurate, so we therefore also add the additional force term, so that we are able to analyse the impact of different forces on the numerical solution, for instance, force over a certian time period, in addittion.  This will allow for further analysis into why our system behaves as it does, and these physical implications. 
# 

# In[14]:


# Verlet's method is the most accurate of the four, for the spring and mass constant being used, as we will see 
# later in the script, so we therefore include a force calculation for this method, hence why it takes the force
# as an input into the function, whereas none of the other methods do. 

def Verlets_Method(nsteps, hstep, DAMPING_CONSTANT, force = None):
    """
    Simulates the motion of a damped harmonic oscillator using Verlet's method.
    
    Parameters:
    nsteps (int): The total number of time steps for the simulation.
    hstep (float): The time increment for each step.
    DAMPING_CONSTANT (float): The damping coefficient of the oscillator.
    force (ndarray, optional): An array representing an external force applied at each time step.
    
    Returns:
    tuple: Two ndarrays representing the displacement and velocity of the oscillator at each time step.
    """
    # Complete the initial setup
    
    displacement, velocity = empty_array(nsteps)
    
    # We put in the initial conditions
    
    displacement[0] = INITIAL_DISPLACEMENT # in m
    velocity[0] = INITIAL_VELOCITY # in m/s
    
    # Since we can't get the next value we must use one of the existing methods to find this
    # Calculate the next displacement and velocity value using the Improved Euler method
    eulers_displacement, eulers_velocity = improved_euler_method(2, hstep, DAMPING_CONSTANT)
    
    displacement[1] = eulers_displacement[1]
    velocity[1] = eulers_velocity[1]
    
    # If no force is specified assume that the system is unforced
    if force is None:
        force = np.zeros(nsteps)
        
        
    # from here we are able to find the rest of the terms in the Verlet's Method as usual 
    
    D = 2 * MASS + DAMPING_CONSTANT * hstep
    A = 2 * (2 * MASS - SPRING_CONSTANT * hstep ** 2) / D
    B = (DAMPING_CONSTANT * hstep - 2 * MASS) / D
    
    # Calculate the displacement at i+2
    for i in range(nsteps - 2):
        displacement[i+2] = A * displacement[i+1] + B * displacement[i] + (hstep ** 2 * force[i+1])
        velocity[i+2] = (displacement[i+2] - displacement[i]) / (2 * hstep)

    return displacement, velocity

displacement, velocity = Verlets_Method(number_of_steps, step_size, DAMPING_CONSTANT)


# In[15]:


plot_times(INTEGRATION_TIME, TIME_STEPS, Verlets_Method)
fit_plot(INTEGRATION_TIME, TIME_STEPS, Verlets_Method, 'yes', __, __)
plot_energy(INTEGRATION_TIME, TIME_STEPS, Verlets_Method)


# ## Verlet Method Analysis
# 
# 
# In the above plot, we're looking at how the Verlet method conserves energy in our oscillator. Because the Verlet method is symplectic, it should conserve energy very well, making it ideal for our long-term simulation of an oscillator.
# 
# 
# The first graph shows the Verlet Method Plot: the Verlet method's plots for varying h values demonstrate its superior performance in simulating the oscillator's motion, particularly over long periods.
# Goodness of Fit and Energy Dissipation: The Verlet method exhibits exceptional energy conservation, with minimal dissipation compared to the Euler methods. This is evident in the energy plot, where the total mechanical energy remains more constant over time.
# 

# 
# ### Graph: Verlet Method Energy Conservation
# 
# In the above plot, we're looking at how the Verlet method conserves energy in our oscillator. Because the Verlet method is symplectic, it should conserve energy very well, making it ideal for our long-term simulation of an oscillator.
# 

# ## Euler-Cromer Method (Semi-Implicit Euler Method)
# 
# This variant of the Euler method calculates the velocity at the end of the interval, which is then used to update the position. It is a simple yet effective modification that can provide improved results for certain types of problems.
# 

# the **Euler-Cromer's Method** is 
# $$x_{i+1} = x_{i} + h \cdot v_{i+1}$$ 
# $$v_{i+1} = v_{i} - h*k/m\cdot x_{i}$$    

# In[16]:


def euler_cromer_method(nsteps, hstep, DAMPING_CONSTANT):
    """
    Computes the motion of a damped harmonic oscillator using the Euler-Cromer method.
    
    Parameters:
    nsteps (int): The number of steps to simulate.
    hstep (float): The time interval between steps.
    DAMPING_CONSTANT (float): The damping constant applied to the oscillator.
    
    Returns:
    tuple: Arrays containing the displacement and velocity after each step.
    """
    # Complete the initial setup
    
    displacement, velocity = empty_array(nsteps)
    
    # We put in the initial conditions
    
    displacement[0] = INITIAL_DISPLACEMENT # in m
    velocity[0] = INITIAL_VELOCITY # in m/s

    # Calculate the acceleration at step i and the displacement and velocity at step i+1
    for i in range(nsteps - 1):
        #acceleration = - (spring_constant / mass) * displacement[i] - (damping_constant / mass) * velocity[i]
        
        velocity[i+1] = velocity[i] - ((hstep * SPRING_CONSTANT) / MASS) * displacement[i]
        displacement[i+1] = displacement[i] + velocity[i+1] * hstep
        
    return displacement, velocity 

displacement, velocity = euler_cromer_method(number_of_steps, step_size, DAMPING_CONSTANT)


# In[17]:


plot_times(INTEGRATION_TIME, TIME_STEPS, euler_cromer_method)
fit_plot(INTEGRATION_TIME, TIME_STEPS, euler_cromer_method, 'yes', __, __)
plot_energy(INTEGRATION_TIME, TIME_STEPS, euler_cromer_method)


# INSERT MARKDOWN DESCRIPTION WHICH ANALYSES EACH GRAPH: 1: EULER METHOD PLOT (ANALYSE FOR EACH H), 2: GOODNESS OF FIT, 3: ENERGY DISSIPATION (ANALYSE FOR EACH H)

# 
# ### Graph: Euler-Cromer Method Energy Conservation
# 
# The Euler-Cromer method graph shows us if this modified Euler method manages to conserve the system's energy more effectively. It's a balance between simplicity and accuracy.
# 

# 
# ## Method Comparison
# 
# We compare the numerical results with the analytical solution to evaluate their accuracy. The analytical solution for our underdamped oscillator is given by:
# 
# $$
# x(t) = e^{-\frac{b}{2m}t} \left( A \cos(\omega_d t) + B \sin(\omega_d t) \right)
# $$
# 
# where $( \omega_d )$ is the damped natural frequency and \( A \), \( B \) are constants determined by initial conditions.
# 
# We expect the Verlet's method to exhibit the most accurate energy conservation due to its symplectic nature. Let's verify this by simulation.
# 

# In[18]:


def plot_fit_comparison(INTEGRATION_TIME, TIME_STEPS, method1,method2,method3,method4):
    
        fig, ax = plt.subplots(figsize=(12, 6))
        methods = [method1,method2,method3,method4]
        colour = ['r','b','k','orange']
        labels = ["Euler's", "Improved Euler's", "Verlet's","Euler Cromer's"]
        
        for i in range(len(methods)):
            fit_plot(INTEGRATION_TIME, TIME_STEPS, methods[i], 'no', colour[i], labels[i])
     
        fig_routine(ax,'Step Size (s)','Goodness of Fit (%)')
        
plot_fit_comparison(INTEGRATION_TIME, TIME_STEPS, euler_method,improved_euler_method,
                    Verlets_Method,euler_cromer_method)


# As we can see from the above graph, where the errors of each method can be seen together, the Verlet's method has an error extremely close to zero the entire time, with the error increasing in small increments as the step size increases considerably. 
# 
# However, with the Euler and Improved Euler methods, it can be seen that the percentage error increases detrementally as the step size increases, meaning it can only be deemed accurate for small step size values.  
# 
# Intrestingly, the Euler Cromer method has a consistent error close to 85%, regardless of the step size.  Whilst this error is alarmingly large, it seems that it does not vary dependent on step size, and there could perhaps be another influential factor.

# It can be seen that Verlet's method is the best method, with the most accurate energy distribution and all values of step spacing causing the distribution to be incredibly close to the analystical solution, along with the errors being the closest to zero and not largley impacted by the increase of step size. 
# 
# Therefore, we can now vary the damping constant and analyse the effects of verlet's method on this as it changes. We will include analyses of double and half of the original damping constant and understand how these change. 
# 
# We can find the critical damping constant from the use of equation $$b_{cr}^2=4km$$

# In[19]:


critical_damping_constant = np.sqrt(4 * SPRING_CONSTANT * MASS)
damping_constants = [critical_damping_constant, critical_damping_constant*2, critical_damping_constant*0.5]

print("For Verlet's Method: ")
print("Time-Displacement Graphs: ")
for value in damping_constants:
    print(f"damping constant {value:.3f}")
    if value == damping_constants[0]:
        print("This value is the critical damping constant")
    DAMPING_CONSTANT = value
    plot_times(INTEGRATION_TIME, TIME_STEPS, Verlets_Method)


# 
# In the case of a damped oscillator with a damping constant exactly equal to the critical damping value, the system does not oscillate but instead returns to equilibrium as quickly as possible without overshooting, as can be seen in the first graph. This is why the time displacement graph for the Verlet method with critical damping does not exhibit a sinusoidal behavior but instead shows an exponential decay towards equilibrium. 
# 
# At twice the critical damping value (overdamping), the system is heavily damped and returns to equilibrium without oscillating. However, the return to equilibrium is slower compared to critical damping. The system still follows an exponential decay back to equilibrium, but the form of the equation is different, and it takes longer for the system to stabilize compared to critical damping.  This can be seen in the second graph, with double the value of the damping constant. 
# 
# For damping less than the critical value (underdamping), the system oscillates with a gradually decreasing amplitude – that's where you'd expect a sinusoidal-like graph, modulated by an exponential decay. At exactly half the critical damping, the system is underdamped and thus oscillates, but the amplitude of oscillation decreases more quickly than it would with very light damping.  This can be seen in the third graph, as we see a small amount of sinusodial motion, and we would expect to see more oscillations with an even smaller damping constant. 

# In[20]:



print("Energy Dissipation Graphs: ")
for value in damping_constants:
print(f"damping constant {value:.3f}")
if value == damping_constants[0]:
    print("This value is the critical damping constant")
DAMPING_CONSTANT = value
plot_energy(INTEGRATION_TIME, TIME_STEPS, Verlets_Method)

DAMPING_CONSTANT = 0.08


# For a damped harmonic oscillator, the energy dissipation is related to the system's ability to do work, which in the case of a mechanical system like a spring-mass-damper, is the work done by the spring force. The energy of such a system typically consists of potential and kinetic energy components, and when damping is present, energy is gradually removed from the system, and then is usually transformed into heat due to friction or resistance.
# 
# The first graph shows the critical damping (= 4.356). The energy dissipates quickly and does not pass through the equilibrium position more than once. This is reflected in the graph by a steep decline in energy that flattens out as the system approaches the equilibrium position.
# 
# The second graph shows 2 times the critical damping constant, where the system is overdamped. The return to equilibrium is much slower than at critical damping, and there are no oscillations. Therefore the energy still dissipates, but more slowly compared to the critically damped case. The curve would show a decline in energy, but less steep than the critical damping curve, reflecting the slower approach to equilibrium.
# 
# The third graph shows 1/2 the critical damping constant, and the system is underdamped. It will oscillate with decreasing amplitude, meaning the energy will also oscillate as it dissipates. However, because the damping is not enough to prevent oscillations altogether, the energy will decrease more gradually compared to critical damping. The curve for energy dissipation shows a more gradual decline and as seen, has a wave-like pattern reflecting the oscillations of the system.
# 

# From the above analyses, it can be seen that as the damping constant varies, the approximation due to Verlet's Method is adjusted appropriatley, with the approximation matching the analystical values for all values of step size, and all values of damping constant.  
# 

# We are now able to explore two scenarios where there is now an oscillatory force, adding this into our most accurate method, which in this case, is the Verlet method. 
# 
# 1. sudden application of an external force after a few oscillation periods (a ‘push’).
# Explore different situations where the force has the same or opposite sign to the instantaneous velocity and is applied in different parts of a cycle. Comment on your findings in the notebook (week 3 of project).
# 
# 
# 2. forced oscillations with a sinusoidal external force with frequency different from the undamped natural frequency.
# Make sure your graph shows steady oscillations after the transient period. Compare with unforced oscillations (use the appropriate comments in the notebook or plots) (week 3 of project).

# We can start by analysing the system where a force has been suddenly applied

# In[32]:


# We want to plot the force for the different methods 

# We start by plotting for the smallest step (since all steps give the same values for the Verlet method) and the
# displacements when we have an external force applied at a certian time

def force_plot(method, step_size, DAMPING_CONSTANT, force=None, label_name=None):
    """
    Plots the displacement of a damped harmonic oscillator under the influence of an external force.
    
    Parameters:
    method (function): The numerical method used for simulation.
    step_size (float): The time step used in the simulation.
    DAMPING_CONSTANT (float): The damping coefficient.
    force (ndarray, optional): An array representing the external force applied at each step.
    label_name (str, optional): The label for the plot legend.
    """
    # Determine total number of steps and define an array of time values
    number_of_steps = math.ceil(INTEGRATION_TIME / step_size)
    time_array = get_time_array(INTEGRATION_TIME, step_size)
    
    # Call and plot the specified method using the specified force and plot label
    displacement, _ = Verlets_Method(number_of_steps, step_size, DAMPING_CONSTANT, force)
    
    # Call and plot the specifed method for an unforced system
    displacement_unforced, _ = Verlets_Method(number_of_steps, step_size, DAMPING_CONSTANT)
    
    
    plt.plot(time_array, displacement, 'r' ,label=label_name)
    plt.plot(time_array, displacement_unforced, 'k--', label=f"Unforced Verlet's Method", lw=1)   
    
    # Set the graph's axes labels and limits
    plt.xlim(0, INTEGRATION_TIME)
    
    # Call the plotting routine
    plt.legend()
    plt.grid(True)
    plot_routine("Time (s)","Displacement (m)")
    

    
def force_instantaneous(time_of_force, size_of_force):
    """
    Creates a force array representing an instantaneous force applied at a specific time.
    
    Parameters:
    time_of_force (float): The time at which the force is applied.
    size_of_force (float): The magnitude of the force.
    
    Returns:
    ndarray: An array with the instantaneous force applied at the specified time.
    """
    # Determine total number of steps and the step at which the force is applied
    step_total = math.ceil(INTEGRATION_TIME / step_size)
    step_force = math.ceil((time_of_force / INTEGRATION_TIME) * step_total)
    
    # Define an array for the force, setting values to zero
    force = np.zeros(step_total)
    
    # Set value at which the force is applied to the force vector
    force[step_force] = size_of_force
    return force


# In[33]:



# Call the plotting function for Verlet's method (step size 0.05s) with an instantaneous force of 100N at 20s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, force_instantaneous(22, 100), '100N at 22s')

# Call the plotting function for Verlet's method (step size 0.05s) with an instantaneous force of -100N at 20s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, force_instantaneous(22, -100), '-100N at 22s')


# Above, we see the forces of 100N and -100N at 22 seconds, which is the approximate position of equilibrium of the system.
# 
# At equilibrium, the forces have the most substantial effect on the amplitude of the wave. We are able to see the massive difference in effects of the 100N and -100N forces at these points. We are also able to interpret that the biggest difference about these newly damped waves due to their position is their amplitude change, where one is much larger, and the other is small, the two waves are exactly in phase and are coherent.
# 
# The -100N force reduces the amplitude of the oscillation, while the 100N force increases it.
# The point of application  is critical because at equilibrium, this is where the system has minimal kinetic energy, and all energy introduced by the force therefore goes into altering the amplitude.
# 
# Adding a force at equilibrium maximizes its effect on amplitude due to the conservation of energy; energy isn't "wasted" on changing kinetic energy but goes directly into potential energy.
# 
# We will also plot below, the forces when applied at maximum amplitude too.  We will be able to see that the forces have their largest effect on the amplitude at equlibrium.

# In[34]:


# Call the plotting function for Verlet's method (step size 0.05s) with an instantaneous force of 100N at 20s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, force_instantaneous(18.5, 100), '100N at 18.5s')

# Call the plotting function for Verlet's method (step size 0.05s) with an instantaneous force of -100N at 20s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, force_instantaneous(18.5, -100), '-100N at 18.5s')


# When we apply the forces of 100N and -100N at the maximum amplitude position, or maximum displacement, we are able to see above that the amplitude of the new decaying wave is the same for both, with the two waves being out of phase, still with a constant phase difference.  Neither wave is in phase with the origional unforced wave. 
# 
# For the -100N force, the wave following the point of force application has a phase shift causing it to start its cycle earlier than the unforced wave. This suggests the force effectively 'pulls' the system back towards equilibrium, accelerating the phase of the oscillation without noticeably changing the amplitude.
# For the 100N force, the phase shift is in the opposite direction. The system 'pushes' forwards, delaying the return to the equilibrium point, thus advancing the phase of the oscillation in the direction of the applied force.
# Phase shifts, rather than amplitude changes, occur because the system is at maximum amplitude; the force is applied when the system's velocity is zero (or near zero), and all the energy from the force goes into shifting the position rather than changing the kinetic or potential energy.
# 

# We can now explore the effect of a continuous force on the system

# In[38]:


def continuous_force(time_force_start, time_force_end, size_of_force, frequency):
    """
    Generates a time-dependent force array for a continuous sinusoidal force applied over a time interval.
    
    Parameters:
    time_force_start (float): The time at which the sinusoidal force begins to be applied.
    time_force_end (float): The time at which the sinusoidal force stops being applied.
    size_of_force (float): The amplitude of the sinusoidal force.
    frequency (float): The frequency of the sinusoidal force.
    
    Returns:
    ndarray: An array representing the sinusoidal force applied at each time step.
    """
    # Determine total number of steps and the step at which the force is applied
    step_total = math.ceil(INTEGRATION_TIME / step_size)
    step_force_start = math.ceil((time_force_start / INTEGRATION_TIME) * step_total)
    step_force_end = math.ceil((time_force_end / INTEGRATION_TIME) * step_total)
    
    
    # Define an array for the force, setting values to zero
    force = np.zeros(step_total)
    
    # Set values over which the force is applied to the force vector
    # This gives us a constant sinusodial force applied to the system
    
    for step, _ in enumerate(force):
        if step > step_force_start and step < step_force_end:
            time = step / step_total * INTEGRATION_TIME
            force[step] = size_of_force * np.sin(frequency * time)
            
    return force
    
    
# Call the plotting function for Verlet's method (step size 0.05s) with a force of 1N * sin(1rad/s * t) 
# from 22s to 32s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, continuous_force(0, 140, 5, 1),
           '$5_{N} ( sin(1_{rad/s} * time))$ , from 0s to 140s')


# Call the plotting function for Verlet's method (step size 0.05s) with a force of 1N * sin(0.1rad/s * t)
# from 20s to 30s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, continuous_force(20, 30, 5, 0.1),
           '$5_N ( sin(0.1_{rad/s} * time))$ , from 20s to 30s')


# Call the plotting function for Verlet's method (step size 0.05s) with a force of 1N * sin(10rad/s * t)
# from 20s to 30s
force_plot(Verlets_Method, step_size, DAMPING_CONSTANT, continuous_force(20, 30, 5, 1),
           '$5_{N} ( sin(1_{rad/s} * time))$ , from 20s to 30s')


# In these graphs, we can see the clear difference in applying a continuous force over a period of time, rather than a single instance. This results in different dynamics compared to the instantaneous force applications we previously discussed.
# 
# In the first graph, we apply a force proportional to $5_N(sin(1_{rad/s}*time)$ from 0 to 140 seconds, this is applied continuously from the start to the end of the simulation. Here, we see the system is driven throughout, resulting in a complex superposition of the forced and natural oscillations, and leading to a modulated wave pattern where the amplitude varies over time, showing beats—a classic interference pattern between two waves of slightly different frequencies.
# 
# The second graph shows a force proportional to  $5_N(sin(1_{rad/s}*time)$ from 20 to 30 seconds is applied. This results in a significant increase in the amplitude of the displacement, peaking at the end of the force application period. This is a typical resonance phenomenon, where the driving frequency of the force matches the natural frequency of the system, leading to a cumulative effect on amplitude.
# 
# In the third graph, a force proportional to In the first graph, we apply a force proportional to $5_N(sin(0.1_{rad/s}*time)$ from 20 to 30 seconds. The effect is much less pronounced than with $0.1_{rad/s}$, as in the second graph, indicating that this driving force does not resonate with the system's natural frequency. We can see a slight increase in amplitude, but it's not as dramatic.
# 

# ## Investigating Resonance in a Damped Harmonic Oscillator
# 
# To explore the phenomenon of resonance in a damped harmonic oscillator, we will simulate the system's response to a range of driving frequencies. Specifically, focusing on how the amplitude of oscillation changes with the frequency of an applied sinusoidal force. 
# 
# ### Defining the System's Natural Frequency
# 
# The natural frequency (\( \omega_0 \)) of an undamped harmonic oscillator is crucial for understanding resonance. It is defined by the equation:
# 
# $$
# \omega_0 = \sqrt{\frac{k}{m}}
# $$
# 
# where:
# - k is the spring constant,
# - m  is the mass of the oscillator.
# 
# ### Implementing a Sinusoidal Driving Force
# 
# Incorporate a sinusoidal driving force into the system's equation of motion:
# 
# $$
# F(t) = F_0 \sin(\omega t)
# $$
# 
# where:
# - \( F_0 \) is the amplitude of the driving force,
# - \( \omega \) is the angular frequency of the driving force,
# - \( t \) is the time.
# 
# ### Running Simulations Over a Range of Frequencies
# 
# To investigate resonance, the system is stimulated for various driving frequencies (\( \omega \)), in small increments.
# 
# ### Measuring the Amplitude of Oscillations
# 
# For each frequency (\( \omega \)), the system is allowed to reach a steady state (where transient effects have diminished) and the amplitude is measured for oscillations, by identifying the peak displacements.
# 
# ### Plotting the Amplitude against the Frequency
# 
# Plotting the measured steady-state amplitudes as a function of the driving frequency (\( \omega \)) is imortant to see the resonance graph. The peak of this plot represents the system's resonance frequency, where the amplitude response is maximized.
# 
# This structured approach enables a methodical exploration of resonance within a damped harmonic oscillator, a phenomenon with widespread implications across physics and engineering disciplines.
# 

# In[29]:


omega_0 = np.sqrt(SPRING_CONSTANT / MASS)  # Natural frequency

# Frequency range around natural frequency
frequencies = np.linspace(0.8 * omega_0, 1.2 * omega_0, 100)

amplitudes = np.zeros(len(frequencies))


def resonance_plot(method, time_step): 
    """
    Plots the resonance curve of a damped harmonic oscillator by simulating its response to a range of frequencies.
    
    Parameters:
    method (function): The numerical method used for the simulation.
    time_step (float): The time increment for each simulation step.
    """
    
    total_steps = math.ceil(INTEGRATION_TIME / time_step)

    for i, omega in enumerate(frequencies):
        force = continuous_force(0, INTEGRATION_TIME, 1, omega)
        displacement, _ = method(total_steps, time_step, DAMPING_CONSTANT, force)
        amplitudes[i] = (max(displacement))
        
    # Plotting
    plt.plot(frequencies, amplitudes, 'o', ms=2, label='resonance data')
    plot_routine('Driving Frequency (rad/s)','Steady-State Amplitude (m)')
    
resonance_plot(Verlets_Method, step_size)


# The graph shows the steady-state amplitude of a system in response to a range of driving frequencies. The result is a typical resonance curve, where the amplitude of the system's oscillation increases as the driving frequency approaches the system's natural frequency.
# 
# The curve peaks around a certain frequency, indicating the point of resonance where the system is most efficiently absorbing energy from the driving force, and thereby resulting in the largest amplitude oscillations.  As mentioned prior, this is the systems natural frequency. Beyond this peak, the amplitude decreases as the driving frequency moves away from the natural frequency, showing less efficient energy transfer.
# 
# The flat point around 0.425 $rad/s$ suggests an anomaly where the amplitude does not change with a slight variation in driving frequency. This could be due to several factors:
# 
# Damping: In a physical system, as the driving frequency approaches the natural frequency, the damping mechanism can become more effective at dissipating energy, thereby leading to a plateau in the amplitude response.
# 
# System Nonlinearity: The system may exhibit non-linear behavior near the resonance frequency. Nonlinearities can cause the system to have a more complex response than the simple linear increase up to the peak.
# 
# Data Resolution: The flat point might be an concequence of the data smoothing algorithm creating the appearance of a plateau.
# 
# Physical Limitation: Experimentally, there could be a physical constraint preventing the amplitude from increasing within this frequency range. This could be structural limits in a mechanical system or saturation effects in an electronic system.
# 
# 

# In[53]:


def investigate_flat_point(method, time_step, damping_constant, spring_constant, mass, integration_time, frequency_start, frequency_end, num_points):
    """
    Investigate the flat point in the resonance curve by increasing frequency resolution around the flat point.
    
    Parameters:
    method (function): The numerical method used for the simulation.
    time_step (float): The time increment for each simulation step.
    damping_constant (float): The damping constant applied to the oscillator.
    spring_constant (float): The spring constant of the oscillator.
    mass (float): Mass of the oscillator.
    integration_time (float): The total time over which the simulation is run.
    frequency_start (float): The starting frequency range for investigation.
    frequency_end (float): The ending frequency range for investigation.
    num_points (int): The number of points in the frequency range for finer resolution.
    """
    
    # Generate frequencies with higher resolution around the flat point
    frequencies = np.linspace(frequency_start, frequency_end, num_points)
    amplitudes = np.zeros(len(frequencies))
    
    # Use a smaller time step for higher accuracy
    time_array = np.linspace(0, integration_time, int(integration_time / time_step))
    
    for i, freq in enumerate(frequencies):
        
        time_array = np.linspace(0, integration_time, int(integration_time / time_step))
        # Define the force function for the current frequency
        force_array = np.sin(freq * time_array)
        # Correct the number of steps to be an integer
        nsteps = int(integration_time / time_step)
        
        # Run the simulation
        displacement = method(nsteps=nsteps, hstep=time_step, DAMPING_CONSTANT=damping_constant, force=force_array)

        # Select the last 10% of the simulation data for steady-state analysis
        steady_state_displacement = displacement[-int(0.1 * len(displacement)):]  # Last 10% of the simulation
    
        #Ensure steady_state_displacement is a 1-D array before getting max and min
        
        # Calculate the peak-to-peak amplitude in the steady state
        amplitudes[i] = np.ptp(steady_state_displacement)
       
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, amplitudes, 'o', ms=4, label='Resonance Data around Flat Point')
    plot_routine('Driving Frequency (rad/s)', 'Steady-State Amplitude (m)')

# Adjust this line according to your actual Verlets_Method signature
investigate_flat_point(Verlets_Method, TIME_STEPS[0], DAMPING_CONSTANT, SPRING_CONSTANT, MASS, INTEGRATION_TIME, 0.41, 0.44, 100)


# The above function refines the investigation around the frequency range where the flat point is observed. By plotting a detailed graph of this region, we can better understand the behavior of the system and whether the flatness is a result of the discrete steps in frequency or an actual characteristic of the system's response. 
# 
# Since the flatness persists even with a high resolution of frequencies, it is likely a real feature of the system and therefore warrants further investigation into the physics of the model or the numerical methods used.  It could also be indicative of a system entering a state of dynamic equilibrium. It could also indicate a system state where the driving force and damping effects balance each other out. meaning that small changes in the driving frequency around this point don't significantly affect the amplitude of the oscillations. 
# 

# 
