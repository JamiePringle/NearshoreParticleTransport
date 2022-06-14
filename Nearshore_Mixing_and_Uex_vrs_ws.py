from pylab import *
from numpy import *
import multiprocessing as mp
import time
from sklearn.neighbors import KernelDensity

# This code calculates and plots the mean horizontal diffusivity and
# horizontal advection of particles that target a range of depths as a
# function of the speed at which the particles move toward a target
# depth, and that target depth.
#
# In the first figure, the top plot shows the horizontal diffusivity
# Kparticle scaled by Kex, the diffusivity it would have if the
# particles had no vertical behavior (i.e. w_s=0). This plot is
# function of the speed with which a particle move towards it target
# depth w_s. This plot is not a function of the target depth.
#
# The bottom plot shows the horizontal velocity of the particle as a
# function of the speed with which it approaches its target depth,
# w_s, and the target depth it seeks, h_t.
#
# The second panel shows the depth distribution and cross-shelf distribution of particles
#
# All units are MKS unless explicitly stated (TcalcDays and TintDays are in days)

#Define the bathymetry
def topo(x):
    #Note that if the depth is constant, the results for mean
    #horizontal transport and mixing are less obvious to interpret,
    #since the particles will be spreading to different depths with
    #different conditions.  this is currently just a constant 20m
    h=20.0+0*x
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
def runModel(ustar,Uex,wSwim,swimTo,xStart,zStart,TintDays,TcalcDays,Np=20000):
    '''ustar and Uex are given to the functions makeVels and mixing to determine the 
    cross shelf velocity and vertical mixing. Please see those functions for details.

    wSwim is the vertical swimming speed of the particles, and swimTo is the depth they will swim to.
    xStart and zStart are the initial horizontal and vertical position of the particles. 

    TintDays is the duration of the model run.
    TcalcDays is the time over which the mean transport and dispersal of the particles is calculated.

    Np=20000 is the number of particles to run with. Defaults to
    20000. More particles leads to less noise in the estimates, but
    longer run time.

    Returns an estimate of the horizontal diffusivity Kx, the mean
    cross-shelf velocity of the particles, Ucrit, and the location of
    the particles after TintDays, xP and zP

    '''

    print('running u*=%3.1e, wSwim=%3.1e, swimTo=%3.1f, '%(ustar,wSwim,swimTo),end='')
    
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
    Ndiff=int(TcalcDays*8.64e4/dt) #over how many time steps to compute mean speed and variance increase
    Nsave=5 #how often to keep data
    tHist=[] #record when variance is calculated
    varHist=[] #keep a history of the variance of particle position in x
    meanHist=[] #keep a history of the variance of particle position in x
    nplotsMade=-1 #for use below
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
        assert dt*abs(wSwim)/topo(0.0)<0.05, 'oops, the vertical jumps are too big'

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

        #do I plot to debug?
        if False:
            Nplot=20
            if remainder(nstep,Nplot)==0:
                nplotsMade+=1
                #erase old points
                if nplotsMade>0:
                    pointHandles[0].remove()
                else:
                    #draw initial locations
                    clf()
                    plot(xPinit/1e3,zPinit,'g*',zorder=10)
                    #and comment on the time scale of mixing
                    print('At a depth of 10m, dt/(h/ustar)=%2.1e. It should be <<1'%(dt/(10.0/ustar),))
                    zPlast=zP.copy()
                    xPlast=xP.copy()

                pointHandles=plot(xP/1e3,zP,'r.',zorder=1,alpha=0.5)
                title(r'$\Psi$ and particle position at t=%4.2f days for $u^*$=%2.1em/s'%(dt*nstep/8.64e4,ustar)
                      +'\n'+r'$w_{swim}$=%2.1e and $w_{swim}/u^*$=%2.1e to depth of %2.1fm'%
                      (wSwim,wSwim/ustar,swimTo),
                      fontsize='large')
                axis(ymin=-topo(0))
                draw()
                show()
                pause(0.01)

        #store history of variance
        tHist.append(dt*nstep)
        varHist.append(std(xP)**2.0)
        meanHist.append(mean(xP))

    #now calculate Kx and Ucrit. Kx=0.5*dVar/dt, Ucrit=dMean/dt
    Kx=0.5*(varHist[-1]-varHist[-Ndiff])/(tHist[-1]-tHist[-Ndiff])
    Ucrit=(meanHist[-1]-meanHist[-Ndiff])/(tHist[-1]-tHist[-Ndiff])

    print('returning Kx=%3.1e, Ucrit=%3.1e'%(Kx,Ucrit),flush=True)#,dt,r,K)


    #return the estimate of the horizontal diffusivity Kx and velocity Ucrit
    #along with the final location of the particles. 
    return Kx,Ucrit,xP,zP
        

#now run
if __name__=="__main__":


    #==================================================================================================
    #make the first panel by evaluating the mean transport and mean dispersal of particles 
    
    fig=figure(1,figsize=[8,11])
    clf()
    style.use('ggplot')

    #what parameters to use. Note, you want a smoother figure, change
    #the True to False below
    if True:
        pi1vec=linspace(0.0,2.0,40//4) #pi1 is w/ustar
        pi3vec=linspace(-0.95,-0.05,11) #pi3=swimTo/h
    else:
        pi1vec=linspace(0.0,2.0,40) #pi1 is w/ustar
        pi3vec=linspace(-0.95,-0.05,21) #pi3=swimTo/h
        
        
    pi1mat,pi3mat=meshgrid(pi1vec,pi3vec)
    

    #dimensionalize
    ustar=0.005*ones(pi1mat.shape)
    h=topo(0.0)*ones(pi1mat.shape)
    wSwim=pi1mat*ustar
    swimTo=pi3mat*h

    #what is the initial horizontal and vertical location of the particles
    xStart=0.0
    zStart=-topo(xStart)/2

    #how long to run the model (TintDays), and how much of the END of
    #the integration period to calculate the diffusivity and mean
    #motion (TcalcDays). Note that TcalcDays and TintDays should be such that
    #the particles have reached their equilibrium depth
    #distribution. But note if you are running this over a sloping
    #bottom, the particles may have moved across the shelf to a
    #different depth.
    #
    #Both in units of DAYS
    TintDays=3.0
    TcalcDays=0.5

    #from Horwitz & Lentz, assume u scales as HL*ustar, where HL about 5.0
    #then u=-HL*2*ustar*(1/2+z/h) for a downwelling flow, and minus that for upwelling
    #and the surface and bottom cross-shelf velocities are about Uex=HL*ustar
    Uex=5.0*ustar

    if True: #if this is false, don't make first plot. Useful for debugging.
        #matrices to hold answers
        KxVec=nan+ustar
        UcritVec=nan+ustar

        #run model
        uSwimTo=swimTo+nan

        #run models with multi processing
        pool=mp.Pool()
        argList=[]
        whereList=[]
        print('starting parallel runs')
        for n0 in range(ustar.shape[0]):
            for n1 in range(ustar.shape[1]):
                argList.append((ustar[n0,n1],Uex[n0,n1],wSwim[n0,n1],swimTo[n0,n1],xStart,zStart,TintDays,TcalcDays))
                whereList.append((n0,n1)) #to put answers where they belong

        #now run model
        tic=time.time()
        output=pool.starmap(runModel,argList)
        print('Took',time.time()-tic,'to finish')

        #now unpack the output
        for n in range(len(output)):
            n0=whereList[n][0]
            n1=whereList[n][1]
            KxVec[n0,n1],UcritVec[n0,n1]=output[n][:2]

        #close pool of multiprocessing processes
        pool.terminate()

        #make non-dimensional answers and plot
        KxScale=(1.0/6.0)*Uex**2*h/kappa/ustar
        pi2mat=KxVec/KxScale
        pi4mat=UcritVec/Uex

        clf()

        #plot diffusivity
        ax1=subplot(2,1,1)
        nMiddle=argmin(abs(pi3vec+0.5))
        plot(pi1vec,pi2mat[nMiddle,:],'r-*')
        xlabel(r'$w_s/u^*_{total}$')
        ylabel(r'$K_{particle}/K_{ex}$')
        title(r'Horizontal diffusivity')

        #plot horizontal velocity as function of depth
        ax3=subplot(2,1,2)
        out=contourf(pi1vec,pi3vec,pi4mat,linspace(-1.0,1.0,21),cmap='seismic')
        colorbar(out,orientation='horizontal',fraction=0.15,shrink=0.5)
        xlabel(r'$w_s/u^*_{total}$')
        ylabel(r'$h_{t}/h$')
        title(r'$u_{particle}/u_{ex}$')

        tight_layout()

        draw()
        show()
        pause(0.1)

        savefig('Panel1.png',dpi=100)

    #==================================================================================================
    #now make second figure, which shows the distribution of particles
    #after TintDays for a release at a depth of zStart and for a set
    #of ratios of swimming speed to ustar given in nonDswimSpeedVec
    figure(2,figsize=[8,11])
    clf()
    style.use('ggplot')
    ax1=subplot(2,1,1)
    ax2=subplot(2,1,2)


    #where do the particles start in vertical, and what are their
    #non-dimensional (Ws/ustar) swimming speeds?
    zStart=-0.25*topo(xStart)
    nonDswimSpeedVec=[0.0,0.1,0.25,0.5,0.75]
    #nonDswimSpeedVec=[0.0,0.5]

    #what depth do the particles swim to?
    swimTo=zStart

    #model velocity parameters
    ustar=0.01
    Uex=5*ustar

    #run model for each swimming speed and store results
    zPall={} #store all zPoints, key is wswim
    xPall={} #store all xPoints, key is wswim
    for wSwim in array(nonDswimSpeedVec)*ustar:
        Kex,Ucrit,xP,zP=runModel(ustar,Uex,wSwim,swimTo,xStart,zStart,TintDays,TcalcDays,Np=40000)

        xPall[wSwim]=xP
        zPall[wSwim]=zP


    #now loop over all results, and find max xP and min xP
    xPmin=Inf
    xPmax=-Inf
    for wSwim in array(nonDswimSpeedVec)*ustar:
        xPmin=min(xPmin,amin(xPall[wSwim]))
        xPmax=max(xPmax,amax(xPall[wSwim]))

    #now plot results
    xPrangeKm=(xPmax-xPmin)/1e3
    for wSwim in array(nonDswimSpeedVec)*ustar:

        labelText=r'$w_s/u^*$=%3.2f'%(wSwim/ustar)
        #print('working on',labelText)

        #plot horizontal distribution. Note that the bandwidth here is worth paying attention to!
        sca(ax1)
        xP=xPall[wSwim]
        xPkm=xP/1e3
        kdeCompute=KernelDensity(kernel='gaussian',bandwidth=0.2).fit(xPkm)
        xPlot=linspace(xPmin/1e3-0.2*xPrangeKm,xPmax/1e3+0.2*xPrangeKm,500)
        logPdf=kdeCompute.score_samples(xPlot.reshape(-1,1))
        plot(xPlot,exp(logPdf),'-',label=labelText)
        #plot(xPkm,0.0*xPkm,'+')

        #plot horizontal distribution. Note that the bandwidth here is
        #worth paying attention to!  note Zignore -- this is the
        #region near the top and bottom where KernelDensity is
        #affected by boundary conditions, and so is not plotted.
        sca(ax2)
        zIgnore=0.5
        zP=zPall[wSwim]
        kdeCompute=KernelDensity(kernel='gaussian',bandwidth=.25).fit(zP)
        zPlot=linspace(-topo(xStart)+zIgnore,-zIgnore,500)
        logPdf=kdeCompute.score_samples(zPlot.reshape(-1,1))
        plot(exp(logPdf),zPlot,'-')

        draw()
        show()
        pause(0.1)

    sca(ax1)
    title('horizontal distribution of particles',fontsize='large')
    jnk=axis(); plot([0.0,0.0],jnk[2:],'k--')
    ylabel('PDF')
    xlabel('cross-shelf distance, km')
    legend()
    
    sca(ax2)
    jnk=axis(); plot(jnk[:2],[swimTo,swimTo],'k--')
    title('vertical distribution of particles',fontsize='large')
    xlabel('PDF. Note distortion near top and bottom due to PDF calculation')
    ylabel('depth, m')

    suptitle(r'Distribution after %2.1f days, $u^*=%3.2f m\,s^{-1}$, $u_{ex}=%3.2f m\,s^{-1}$,'%(TintDays,ustar,Uex)+
             '\n'+r'swimming to %2.1f m. '%(swimTo)+
             'Dashed line is horizontal starting position\nor target depth',fontsize='x-large')
    
    draw()
    show()
    savefig('Panel2.png',dpi=100)

  



    
