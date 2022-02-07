
import numpy as np
from matplotlib import pyplot as plt


def getpolicy(Q):
    """ GWGETPOLICY
    Get best policy matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """

    P = None

    return P


def getvalue(Q):
    """ GWGETVALUE
    Get best value matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """
    
    V = None

    return V

def plotarrows(P):
    """ PLOTARROWS
    Displays a policy matrix as an arrow in each state.
    """

    x,y = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))

    u = np.zeros(x.shape)
    v = np.zeros(y.shape)

    v[P==2] = 1
    v[P==3] = -1
    u[P==0] = -1
    u[P==1] = 1

    plt.quiver(v,u,color='r')
