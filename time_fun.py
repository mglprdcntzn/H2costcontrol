import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

from tabulate import tabulate
#############################################################
def NRI(barY, h0, barS, barYload, Vinic, itmax, prec):
    #Newton Raphson method for power flow
    NPQ     = Vinic.shape[0]
    unos    = np.ones((NPQ, 1))
    I       = np.eye(NPQ)
    B       = np.conj(barY + barYload)
    invB    = np.linalg.inv(B)
    barSvec = barS@unos
    
    A     = barS@np.linalg.inv(Vinic**2)
    F     = invB@A
    H     = np.linalg.inv(I - F@np.conj(F))@invB
    
    #loop
    ve      = Vinic@unos    
    permiso = True
    itcont  = 0
    while permiso:
        h     = h0 + B@np.conj(ve) - (1/ve)*barSvec
                
        A     = barS@np.diag(np.squeeze(1/ve**2))
        F     = invB@A
        # H     = np.linalg.inv(I - F@np.conj(F))@invB
            
        Hh    = H@h
        Delta = np.conj(F)@Hh - np.conj(Hh)
        
        ve    = ve + Delta
      
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(h, ord=2) < prec:
            permiso = 0
            
    return  np.squeeze(ve)


#############################################################
def reactive_power(P, fp):
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(P))])
    Q = P * np.sqrt(np.reciprocal(fp**2) - 1) * noise
    return Q

#############################################################
def built_profiles_models(filename):
    df       = pd.read_csv(filename)
    num_col  = df.shape[1]
    model    = []
    
    time     = df.iloc[:, 0].to_numpy()
    time     = 60*np.append(time,24) #day time in minutes
    for ii in range(1,num_col):
        col = df.iloc[:, ii].to_numpy()
        col = np.append(col,col[0])
        
        params = []
        
        for per in range(24):
            if per==0:
                tantant = -60
            else:
                tantant = time[per-1]
            yantant = col[per-1]
            
            tant = time[per]
            tpos = time[per+1]
            
            yant = col[per]
            ypos = col[per+1]
            
            if per<23:
                tpospos = time[per+2]
                ypospos = col[per+2]
            else:
                tpospos = 25*60
                ypospos = col[0]
                
            yy = np.array([[yantant],
                           [yant],
                           [ypos],
                           [ypospos]])
            xx = np.array([[tantant],
                           [tant],
                           [tpos],
                           [tpospos]])/(24*60)
            
            flat_xx = xx.flatten()

            # Create the matrix
            XX = np.column_stack((
                flat_xx**3,  # First column: cubic values
                flat_xx**2,  # Second column: square values
                flat_xx,     # Third column: original values
                np.ones(flat_xx.shape)  # Fourth column: ones
            ))
            
            params.append(np.linalg.inv(XX)@yy)
            
        model.append(params)
    return model
#############################################################
def model_interpole(model, tt):
    hr = math.floor(tt / 60)
    while hr > 23:
        hr = hr - 24
        tt = tt - 24 * 60
    while hr<0:
        hr = hr + 24
        tt = tt + 24 * 60
    
    coefs    = model[hr]
    inst     = tt / (24 * 60)
    times    = np.array([[inst**3], [inst**2], [inst], [1]])
    interpol = coefs.T @ times
    
    # noise = np.array([random.gauss(1, 0.002) for _ in range(len(interpol))])
    # interpol = interpol * noise
    interpol = np.clip(interpol, 0, None)  #eliminate negatives
    
    return interpol

#############################################################
def settling_time(P0, P0ref, dt=1.0, band=0.025):
    """
    Compute the settling time of a signal.

    The settling time is defined as the first instant after which
    the signal remains permanently within the tolerance band.

    Parameters
    ----------
    P0 : np.ndarray
        Signal values at successive time instants.

    P0ref : float
        Reference/final value.

    dt : float
        Time step between samples.

    band : float
        Relative tolerance band.
        Default = 0.025 (2.5%)

    Returns
    -------
    ts : float
        Settling time.

    idx : int
        Index corresponding to settling time.

    lower, upper : float
        Bounds of the settling band.
        
        
    function by ChatGPT
    """

    P0 = np.asarray(P0).flatten()

    # Tolerance band
    lower = P0ref * (1 - band)
    upper = P0ref * (1 + band)

    # Check from each point onward
    for i in range(len(P0)):

        remaining_signal = P0[i:]

        if np.all((remaining_signal >= lower) &
                  (remaining_signal <= upper)):

            ts = i * dt
            return ts, i, lower, upper

    # Signal never settles
    return np.infty, np.infty, lower, upper