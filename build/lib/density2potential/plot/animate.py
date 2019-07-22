import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.ticker import AutoMinorLocator
import numpy as np

"""
Animates the time-dependent quantities
"""

def animate_function(params,function,timestep,fig_name,function_name):

    # Define settings for the mp4 that will be saved
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me', bitrate=1000))

    L = 0.5*params.space
    x = np.linspace(-L,L,params.Nspace)

    # Generate figure
    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='both', color='#D3D3D3', fillstyle='full')
    function_plot, = ax.plot([],[],'#0000CD',label='{}'.format(function_name))
    #function_plot2, = ax.plot([],[],'-r',label='{}'.format(function_name))
    iteration = ax.text(0.02,1.03,'',transform=ax.transAxes)
    ax.legend()

    # Figure settings
    def init():
        function_plot.set_data([],[])
        #function_plot2.set_data([],[])
        iteration.set_text('')
        ax.set_ylim(np.amin(function),np.amax(function))
        ax.set_xlim(np.amin(x),np.amax(x))
        ax.set_xlabel('position (a.u)')
        plt.rcParams.update({'font.size':14})

        return iteration, function_plot,# function_plot2,

    # Function (array) to be animated with the index to animate over (time)
    def animate(time):
        #function_plot2.set_data(x,function[time,:])
        function_plot.set_data(x,function[time,:])
        iteration.set_text('Iteration: {0}, time = {1} a.u'.format(time,round(time*params.dt,1)))

        return iteration, function_plot,# function_plot2,

    # Array defining the time steps that are sampled
    time_sampling = np.arange(0,params.Ntime,timestep)

    # Save the animation
    ani = FuncAnimation(fig, animate, time_sampling, init_func=init, interval=200, blit=False)
    ani.save('{0}.mp4'.format(fig_name), writer=writer, dpi=250)


def animate_two_functions(params,function1,function2,timestep,fig_name,function1_name,function2_name):

    # Define settings for the mp4 that will be saved
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me', bitrate=1000))

    L = 0.5*params.space
    x = np.linspace(-L,L,params.Nspace)

    # Generate figure
    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='both', color='#D3D3D3', fillstyle='full')
    function1_plot, = ax.plot([],[],'#0000CD',label='{}'.format(function1_name))
    function2_plot, = ax.plot([],[],'-r',label='{}'.format(function2_name))
    iteration = ax.text(0.02,1.03,'',transform=ax.transAxes)
    ax.legend()

    # Figure settings
    def init():
        function1_plot.set_data([],[])
        function2_plot.set_data([],[])
        iteration.set_text('')
        ax.set_ylim(np.amin(function1),np.amax(function1))
        ax.set_xlim(np.amin(x),np.amax(x))
        ax.set_xlabel('position (a.u)')
        plt.rcParams.update({'font.size':14})

        return iteration, function1_plot, function2_plot,

    # Function (array) to be animated with the index to animate over (time)
    def animate(time):
        function2_plot.set_data(x,function2[time,:])
        function1_plot.set_data(x,function1[time,:])
        iteration.set_text('Iteration: {0}, time = {1} a.u'.format(time,round(time*params.dt,1)))

        return iteration, function1_plot, function2_plot,

    # Array defining the time steps that are sampled
    time_sampling = np.arange(0,params.Ntime,timestep)

    # Save the animation
    ani = FuncAnimation(fig, animate, time_sampling, init_func=init, interval=200, blit=False)
    ani.save('{0}.mp4'.format(fig_name), writer=writer, dpi=250)


