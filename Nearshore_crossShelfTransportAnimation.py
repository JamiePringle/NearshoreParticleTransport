from pylab import *
from numpy import *
import multiprocessing as mp
import time
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import KernelDensity

# This code animates the progress of particles across a nearshore
# region given a cross-shelf velocity profile, a vertical mixing
# profile, and a defined depth keeping behavior.
#
# All units are MKS unless explicitly stated (TcalcDays and TintDays are in days)

#define the parameters of the model run
ustar=1.0e-2  #the strength of the turbulent velocities, m/s
Uex=-5*ustar   #the exchange velocity; this matches the estimate of Horowitz & Lentz for cross-shelf winds in nearshore
wSwim=0.75*ustar #the speed of vertical swimming/floating towards target depth, m/s
swimTo=-15.0 #the target depth of depth seeking behavior, must be less than zero, meters
xStart=10e3 #the initial distance offshore, meters
zStart=swimTo #the initial depth, meters
TintDays=10.0 #the number of days to run the simulation
xToPlot=12e3 #the offshore extent of plotting, in meters

#Define the bathymetry
def topo(x):
    #Note that if the depth is constant, the results for mean
    #horizontal transport and mixing are less obvious to interpret,
    #since the particles will be spreading to different depths with
    #different conditions.  this is currently just a constant 20m
    h=2.0+5.0e-3*x
    return h

#define vertical mixing coefficients as function of u*, water depth h,
#and depth z. Return both the mixing coefficient K and its vertical
#derivative Kz. The units of K are length/time^2, and Kz 1/time^2
kappa=0.41 #von Karmen's constant
def mixing(z,h,ustar):
    K=kappa*ustar*(-z-z**2/h) #from Signell et al 1990 with u*_bottom=u*_top
    Kz=kappa*ustar*(-1-2*z/h) #at z=0, is -kappa*ustar, at z=-h is kappa*ustar
    return K,Kz

#Define cross-shelf stream function as function of Uex, z and h.
#The units of Psi length^2/time. This is where you would alter the
#spatial structure of the velocity.
def Psi(z,h,Uex):
    psi=2.0*Uex*(h/2.0*z+(z**2)/2.0)/h
    return psi

#calculate velocity at x,z point from Psi and return u,w
def makeVels(x,z,Uex):
    #calculate velocity with centered differences, using u=Psi_z, w=-Psi_x
    dx=50.0 #m, these are used only for calculating derivatives
    dz=0.1 #m   
    h=topo(x)
    u=(Psi(z+0.5*dz,h,Uex)-Psi(z-0.5*dz,h,Uex))/dz
    
    hp=topo(x+0.5*dx)
    hm=topo(x-0.5*dx)
    w=-(Psi(z,hp,Uex)-Psi(z,hm,Uex))/dx
    
    return(u,w)

#determine the trajectory of a set of particles
if True:
    # '''ustar and Uex are given to the functions makeVels and mixing to determine the 
    # cross shelf velocity and vertical mixing. Please see those functions for details.

    # wSwim is the vertical swimming speed of the particles, and swimTo is the depth they will swim to.
    # xStart and zStart are the initial horizontal and vertical position of the particles. 

    # TintDays is the duration of the model run.
    # TcalcDays is the time over which the mean transport and dispersal of the particles is calculated.

    # Returns an estimate of the horizontal diffusivity Kx, the mean
    # cross-shelf velocity of the particles, Ucrit, and the location of
    # the particles after TintDays, xP and zP

    # '''

    print('running u*=%3.1e, wSwim=%3.1e, swimTo=%3.1f, '%(ustar,wSwim,swimTo),end='')
    
    #ok, set up particle positions to calculate with
    Np=20000 # number of particles

    #set the particles initial conditions
    xP=zeros((Np,1))+xStart #release at xStart
    zP=zeros((Np,1))+zStart #release at zStart

    xPinit=xP.copy()
    zPinit=zP.copy()

    #now integrate and plot particle positions for Nstep steps of length
    #dt when choosing timestep, make sure h/ustar for smallest h you care
    #about is much greater than the timestep. In general, if this is true,
    #the advection timestep is not a big deal. 
    dt=30.0 #seconds
    N=int(TintDays*8.64e4/dt) #how many model steps to take, so model duration is dt*N
    #Ndiff=int(TcalcDays*8.64e4/dt) #over how many time steps to compute mean speed and variance increase
    Nsave=5 #how often to keep data
    tHist=[] #record when variance is calculated
    varHist=[] #keep a history of the variance of particle position in x
    meanHist=[] #keep a history of the variance of particle position in x
    nplotsMade=-1 #for use below
    nFrame=-1 #frame counter for animation
    for nstep in arange(N):
        #print('doing time',dt*nstep,'seconds')

        #simple runge-kutta second order
        uP,wP=makeVels(xP,zP,Uex) #half step
        xPguess=xP+0.5*dt*uP
        zPguess=zP+0.5*dt*wP

        #now full step with vels evaluated at half step
        uP,wP=makeVels(xPguess,zPguess,Uex)
        xP=xP+dt*uP
        zP=zP+dt*wP

        if True: #include a random walk from mixing
            #now, following Visser 1997 equation 6, add a random walk to the
            #particles. use a uniform random deviate to avoid the outliers
            #associated with a gaussian, and trust in the central limit theorm
            jnk,Kz=mixing(zP,topo(xP),ustar) #compute vertical deriv of K at particle location
            K,jnk=mixing(zP+0.5*dt*Kz,topo(xP),ustar) #compute K at an offset location
            R=random.uniform(low=-1.0,high=1.0,size=zP.shape) #random uniform number from -1 to 1
            r=0.3 #the expected of std(R**2)
            zP=zP+Kz*dt+R*sqrt(2*dt/r*K)
            #assert amin(K)>0,'K<0?'+str(K)

        #now add a swimming velocity, after checking a CFL like criteria
        #make sure we resolve swiming if depth is Hswim or greater
        Hswim=max(10.0,topo(0.0))
        assert dt*abs(wSwim)/Hswim<0.05, 'oops, the vertical jumps are too big. Reduce timestep'

        #smooth w profile with a tanh
        dzSmth=0.5
        wSwimVec=wSwim*tanh(-(zP-swimTo)/dzSmth)
        zP=zP+wSwimVec*dt

        #now, because random steps sometimes leap out of the water, or
        #tunnel into the rock, lets make sure the positions are not being
        #naughty, and put the particles back in the water...
        #print(zP[0],wP[0],wSwim*dt,end='==>')
        hIn=0.1/10 # how far from boundary to put the lost...
        zP[zP>0.0]=-hIn
        indx=zP<-topo(xP)
        zP[indx]=-topo(xP[indx])+hIn
        #print(zP[0])

        #make an animation
        if True:
            Nplot=20*8 #plot every Nplot steps
            N2cull=50 #only plot every N2cull particles
            
            if remainder(nstep,Nplot)==0:
                nplotsMade+=1
                #erase old points
                if nplotsMade>0:
                    #remove points
                    pointHandles[0].remove()

                    #remove histograms
                    for j in vertHistHan:
                        j.remove()
                    for j in horizHistHan:
                        j.remove()

                else:
                    #this is the first time a plot has been made, set up figure
                    fig=figure(1,figsize=(9,9))
                    clf()
                    style.use('ggplot')
                    gs=GridSpec(nrows=4,ncols=4)
                    axMain=fig.add_subplot(gs[:3,:3])
                    axVert=fig.add_subplot(gs[:3,3])
                    axHoriz=fig.add_subplot(gs[3,:3])

                    sca(axMain)
                    #draw initial locations
                    plot(xPinit/1e3,zPinit,'g*',zorder=10)
                    #and comment on the time scale of mixing
                    print('At a depth of 10m, dt/(h/ustar)=%2.1e. It should be <<1'%(dt/(10.0/ustar),))
                    zPlast=zP.copy()
                    xPlast=xP.copy()

                    #plot bathymetry
                    x2plotVec=linspace(0,xToPlot,20)
                    plot(x2plotVec/1e3,-topo(x2plotVec),'k-',linewidth=2)

                    #plot streamfunction
                    x2plotVec=linspace(0,xToPlot,30)
                    z2plotVec=linspace(-1.0,0.0,30) #this is not z yet, see below
                    topoVec=topo(x2plotVec)
                    x2plotMat,z2plotMat=meshgrid(x2plotVec,z2plotVec)
                    z2plotMat=z2plotMat*topoVec #use broadcasting to make correct z
                    h2plotMat=0*z2plotMat+topoVec #also broadcast
                    psiMat=Psi(z2plotMat,h2plotMat,Uex)
                    contour(x2plotMat/1e3,z2plotMat,psiMat,10,colors='k',alpha=0.3)

                #make main plot
                sca(axMain)
                pointHandles=plot(xP[::N2cull]/1e3,zP[::N2cull],'r.',zorder=1,alpha=0.5)
                title(r'Streamfunction $\Psi$ and particles (red) at t=%4.2f days for $u^*$=%2.1em/s'%(dt*nstep/8.64e4,ustar)
                      +'\n'+r'$w_{swim}$=%2.1e and $w_{swim}/u^*$=%2.1e to depth of %2.1fm'%
                      (wSwim,wSwim/ustar,swimTo),
                      fontsize='large')
                axis(ymin=-topo(xToPlot),ymax=0,xmin=0.0,xmax=xToPlot/1e3)
                ylabel('depth, meters')

                #make vertical particle distribution
                sca(axVert)
                #cla()
                vertBins=linspace(-h2plotMat.max(),0.0,100)
                jnk1,jnk2,vertHistHan=hist(zP,vertBins,density=True,orientation='horizontal',color='r')
                title('Vertical Distribution',fontsize='large')
                axis(xmax=amax(jnk1)*1.1)

                #make horizontal particle distribution
                sca(axHoriz)
                #cla()
                horizBins=linspace(0.0,x2plotMat.max()/1e3,200)
                jnk1,jnk2,horizHistHan=hist(xP/1e3,horizBins,density=True,orientation='vertical',color='r')
                xlabel('Horizontal Distribution, kilometers',fontsize='large')
                axis(ymax=amax(jnk1)*1.1)
                
                draw()
                show()
                pause(0.01)

                if False: #save frames in directory animateFrames to make an animated gif
                    nFrame=nFrame+1
                    print('Saving frame %d to animate'%(nFrame,))
                    savefig('animateFrames/frame_%5.5d.png'%(nFrame,),dpi=75)
                    
                #assert False,'as'

        #store history of variance
        tHist.append(dt*nstep)
        varHist.append(std(xP)**2.0)
        meanHist.append(mean(xP))


        


  



    
