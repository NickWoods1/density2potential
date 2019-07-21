import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np

"""
Animates the time-dependent quantities
"""

def animate_function(params,function,timestep,fig_name):

    # Define settings for the mp4 that will be saved
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1000)


    L = 0.5*params.space
    x = np.linspace(-L,L,params.Nspace)
    #x = np.arange(-params.space/2, params.space/2,params.Nspace)
    #x = np.arange(-5,5,params.dx)

    # Generate figure
    fig, ax = plt.subplots()
    function_plot, = ax.plot(x,np.sin(x),'-b')

    # Function (array) to be animated with the index to animate over (time)
    def animate(time):
        function_plot.set_ydata(function[time,:])

    # Figure settings
    def init():
        ax.set_ylim(np.amin(function),np.amax(function))
        plt.rcParams.update({'font.size':14})

        return function_plot,

    # Array defining the time steps that are sampled
    time_sampling = np.arange(0,params.Ntime,timestep)

    # Save the animation
    ani = FuncAnimation(fig, animate, time_sampling, init_func=init, interval=50, blit=False)
    ani.save('{0}.mp4'.format(fig_name), writer=writer)