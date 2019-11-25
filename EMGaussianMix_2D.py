import numpy as np
import matplotlib.pyplot as plt   
import scipy.stats as ss
import imageio
from matplotlib.patches import Ellipse

def randomGen(mean,SD,pis,N):
    #This function generated GMM samples based on proved means, cov's, and Pi's
    #array of samples to be returned
    gen = np.zeros((N,2))
    #random array to select from which gaussian the smaple is taken
    sourcedist = np.random.uniform(0,1,N)
    for x in range(N):
        #these ifs choose which gaussian to take sample from based on Pis
        if sourcedist[x] < pis[0]:
            gen[x] = np.random.multivariate_normal(mean[0],SD[0])  
        elif sourcedist[x] < pis[0] + pis[1]:
            gen[x] = np.random.multivariate_normal(mean[1],SD[1])
        else:
            gen[x] = np.random.multivariate_normal(mean[2],SD[2])
    return gen

Colors = ["red","green","blue","black"] #quick reference for plotting later

def ellipsePlot(mean,cov,dash = None):
    #Function to plot ellipses
    ell = []
    for x in range(3):
        lambda_, v = np.linalg.eig(cov[x]) #calculates eigen values
        lambda_ = np.sqrt(lambda_)
        #matplot lib function to plot ellipse for e-vals and vects:
        #note that width and height are those of an ellipse 2 times that of the cov matrix bc it's prettier
        temp = Ellipse(xy = mean[x],width = lambda_[1]*4,height = lambda_[0]*4, angle = np.rad2deg(np.arccos(v[0, 0])))
        temp.set_facecolor('none')
        temp.set_edgecolor(Colors[x])
        if(dash):
            #dasahes the ellipse if flag is true
            temp.set_linestyle('--')
        temp.set_linewidth(5)
        ell.append(temp)
    return ell

def myNorm(m_new,m_old,v_new,v_old,pi_new,pi_old):
    #norm for convergence, defined as such by convenience as a simple sum of nrm differences between new and old
    norm = 0.0
    for x in range(3):
        norm += np.linalg.norm(m_new[x] - m_old[x])
        norm += np.linalg.norm(v_new[x] - v_old[x])
        norm += np.linalg.norm(pi_new[x] - pi_old[x])
    return norm

def main():
    #Parameters
    N = 500
    mean = [np.array([1,3]),np.array([3,5]),np.array([6,1])]
    Cov = []
    #loop to add some Cov variety, quickly
    for i in range(3):
        Cov.append(np.array([[1+i/10,-.5+i/10],[-.5+i/10,1-i/10]]))
    pis = [.6,.15,.25]
    #Data Generation
    data = np.full((N,3),3.0)
    data[:,:2] = randomGen(mean,Cov,pis,N) #last column will be meant for color spec (3 is 'black' which is default)
    #Plotting Parameters
    plt.ion()       #turns on dyanmic plotting for gif creation 
    fig = plt.figure(figsize = (12,8)) 
    ax = fig.add_subplot(111)
    filenames = []              #gif files storage
    ell = ellipsePlot(mean,Cov,dash = 1)
    
    #Initialization of EM parameters:
    muK = [np.array([3,-1]),np.array([4,7]),np.array([5,2])]
    varK = [np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])]
    PiK = np.array([.33,.33,.34])
    GammaNK = np.zeros((3,N))
    Nk = np.zeros(3)
    den = None      #temp variable for denominator for each gamma_znk calc

    #Just dummy old values for while loop convergence
    oldmuK = [np.array([0,0]),np.array([0,0]),np.array([0,0])]
    oldvarK = [np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]])]
    oldPiK = PiK - 1
    eps = .01 #Arbitrary, mostly just meant so that convergence is reasonably quick
    iteration = 0

    while(1):
        #Do plotting and gif creation things
        for i in range(3):   
            ax.add_artist(ell[i])   #Draws in correct ellipses
        for n in range(N):
            #plot each point by color (why that 3rd column in the data field exists):
            ax.plot(data[n,0],data[n,1],'o',color = Colors[int(data[n,2])])
        Newell = ellipsePlot(muK,varK)
        for i in range(3):   
            ax.add_artist(Newell[i]) #draw in EM ellipses
        fig.canvas.draw()       #update figure
        filetitle = "ExpMax_2D_" + str(iteration) + ".png"   
        fig.savefig(filetitle)
        filenames.append(filetitle)       #Append to list of file names for gif
        iteration += 1
    
        
        
        #printing so the program spits out updates each step:
        print("Mu", muK,"Var", varK, "Pi", PiK, "eps",myNorm(muK,oldmuK,varK,oldvarK,PiK,oldPiK))
        #Check if Converged:
        if(myNorm(muK,oldmuK,varK,oldvarK,PiK,oldPiK) < eps):
            break
        ax.clear()  #clears axes for next iteration
        #Store Old Values
        oldmuK = muK
        oldvarK = varK
        oldPiK = PiK
        Nk = np.zeros(3) #reset Nk every step
        #Calculate Gamma(Z_NK) for the E step
        for k in range(3):
            for n in range(N):
                den = 0
                for j in range(3):
                    den += PiK[j]*ss.multivariate_normal(muK[j],varK[j]).pdf(data[n,:2])  
                GammaNK[k,n] = PiK[k]*ss.multivariate_normal(muK[k],varK[k]).pdf(data[n,:2])/den
                #Calculate NK while I'm at it
                Nk[k] += GammaNK[k,n] #Application of Eq 9.27

        #M-step continues here (already have Nk from efficiency time save earlier)
        PiK = Nk/N  #Eq 9.26
        muK = [np.zeros(2),np.zeros(2),np.zeros(2)] #reset muK
        varK = [np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2))] #reset varK 
        #Eq 9.24;
        for k in range(3):
            for n in range(N):
                muK[k] += GammaNK[k,n]*data[n,:2]
            muK[k] = np.divide(muK[k],Nk[k])
        #Eq 9.25
        for k in range(3):
            for n in range(N):
                #matrix multiplying to get covariance
                varK[k] += GammaNK[k,n]* (data[n,:2] - muK[k]).reshape(-1,1) @ \
                                         (data[n,:2] - muK[k]).reshape(1,-1)
            varK[k] = np.divide(varK[k],Nk[k])
        #Reclassify point color to the 'k' where 'responsibility' is max for each xn
        for n in range(N):
            data[n,2] = np.argmax(GammaNK[:,n])
    #Compile Images into gif and save (comment out if you dont want gif)
    with imageio.get_writer('ExpMax_2D.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

if __name__ == "__main__":
    main()