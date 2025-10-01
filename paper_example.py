import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cp

import circuit_fun as ct
import time_fun as tm

from scipy.linalg import expm

from matplotlib.ticker import FixedLocator

import random

########################################################
#Circuit generator
N = 50
Dmin  = 250  #min distance btwn nodes
Dmean = 500  #max distance btwn nodes
V = 66  #in kV
NPV   = 1 #regulation nodes
NPQ   = N - NPV

nodes, lines = ct.create_tree_circuit(N, Dmean, Dmin)
lines        = ct.add_loops_to_circuit(nodes, lines, Dmean, Dmin)
########################################################
#nodes classification (regulation, solar, hydrogen, load)
Nsol          = 15
Nhid          = 17

#choose which nodes are reg, sun, hid, and lds 
indexes_reg, indexes_hid, indexes_sol, indexes_lds = ct.choose_nodes(nodes, lines, NPV, Nsol, Nhid)
#reorder nodes to have NPV first
nodes, lines, indexes_reg,indexes_sol,indexes_hid,indexes_lds = ct.reorder_nodes(nodes, lines, indexes_reg,indexes_sol,indexes_hid,indexes_lds)

Nlds         = len(indexes_lds)
########################################################
#load DG and load profiles
Nprofiles = 3 #number of profiles of each type (load, sun, wind)
models    = tm.built_profiles_models('perfiles.csv')
sigma2    = 0.01

fpLoadmax = 0.95
fpLoadmin = 0.75
loadfpmin = 0.4
mfp       = (fpLoadmax - fpLoadmin)/(1- loadfpmin)
########################################################
ST = 1200 #[kVA] rate trafos
S  =  ST*(Nlds+Nhid)#ST*Nlds#ST*(N-NPV)#
########################################################
#loads at nodes
load, loadmix = ct.load_circuit(Nlds, 4*ST/4, 0.01*ST,Nprofiles) #in kW
pv            = ct.DG_circuit(Nsol, 2*ST/4, 0.01*ST)  #in kW
hid           = ct.DG_circuit(Nhid, 4*ST/4, 0.01*ST)  #in kW

installed              = np.zeros((N,1))
installed[indexes_lds] = load.reshape((Nlds,1))
installed[indexes_sol] = pv.reshape((Nsol,1))
installed[indexes_hid] = hid.reshape((Nhid,1))
installed              = installed/S

profilesatnodes                = np.zeros((N,Nprofiles))
profilesatnodes[indexes_lds,:] = loadmix.reshape((Nlds,Nprofiles))

#impendances of the circuit
Y, Y0, Y00    = ct.impendances_circuit(lines, N, NPV)
Ybase         = S / (V**2) / 1000  #divide by 1000 to obtain Ybase in Ohms
########################################################
#normalized circuit
barY    = Y / Ybase
barY0   = Y0 / Ybase
barY00  = Y00 / Ybase

barLoad = load / S
barpv   = pv / S

diagbarY0 = np.diag(barY0.flatten())
########################################################
ct.print_circuit(nodes, lines, 'example_circuit',indexes_reg,indexes_sol,indexes_hid,indexes_lds)
########################################################
#define time
t0 =-0.0*24*60  #begining of time in min
tf = 1.00*24 * 60  #end of time
T  = 1#0.5  #simulation time step
nn = int((tf - t0) / T) + 1  #num of instants
t = np.linspace(t0, tf, nn)  #time vector in mins
########################################################
#drawing constants
hrsperline    = 4
custom_ticks  = np.arange(0, t[-1]/60+1, hrsperline)
minor_ticks   = np.arange(0, t[-1]/60+1, 1)

custom_labels = []
for tick in custom_ticks:
    if tick==0:
        custom_labels = custom_labels + ['00:00']
    elif tick%24==12:
        custom_labels = custom_labels + ['12:00']
    elif tick%24==0:
        custom_labels = custom_labels + ['24:00']
    else:
        custom_labels = custom_labels + ['']
        
########################################################
#NRI0 algorithm params
itmax = 25
prec = 0.00001
########################################################
#initial conditions for NR iterations on PQ nodes
R0    = np.eye(NPQ)
Phi0  = np.zeros((NPQ, NPQ))
Vinic = R0 @ expm(1j * Phi0)
########################################################
#Control periods
T0   = 10 #time for Ctrl of Regulation nodes
n0   = np.floor(T0/T)
ndev = 2

ctrlsteps0 = np.random.randint(max(1,n0-ndev),n0+ndev,NPV)
delay0     = np.random.randint(1,n0,NPV)
T0real     = ctrlsteps0.reshape((NPV,1))*T

T1   = 2 #time for Ctrl of H2 nodes
n1   = np.floor(T1/T)
ndev = 2

ctrlsteps1 = np.random.randint(max(1,n1-ndev),n1+ndev,Nhid)
delay1     = np.random.randint(1,n1,Nhid)
T1real     = ctrlsteps1.reshape((Nhid,1))*T
########################################################
#Market constants
pH2  = 5
########################################################
#Regulation control gains.
P0ref  = 1.00*np.sum(installed[indexes_lds])

eant    = np.zeros((NPV,1))
eantant = np.zeros((NPV,1))

K = 3
L = 24

c = 1/60

########################################################
#H2 Control gains
etamin  = 0.40
eta     = etamin + (1-etamin)*np.random.rand(Nhid)

longeta = np.zeros((N,1))
longeta[indexes_hid] = eta.reshape(Nhid,1)
longeta = longeta[NPV:]

########################################################
# H2 linear cost
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Linear H2 cost")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#prefill vectors and matrices
ve        = np.zeros((NPQ, nn), dtype=complex) #voltage at PQ nodes
ve0       = np.ones((NPV, nn), dtype=complex)  #voltage at sources
ve0_ref   = np.ones((NPV, 1), dtype=complex)

barS0     = np.zeros((NPV, nn), dtype=complex)

P_Sun     = np.zeros((NPQ, nn))
P_lds     = np.zeros((NPQ, nn))
P_Hyd     = np.zeros((NPQ, nn))
Q_Sun     = np.zeros((NPQ, nn))
Q_lds     = np.zeros((NPQ, nn))
Q_Hyd     = np.zeros((NPQ, nn))

fp_Sol_ref = 0.99*np.ones((NPQ, 1))

baryLoad   = np.zeros((NPQ, nn), dtype=complex)

loadsprofiles  = np.zeros((Nprofiles, nn))
SunIrrprofiles = np.zeros((Nprofiles, nn))

P_Hyd_ref  = np.zeros((NPQ, 1))
fp_Hyd_ref = 0.99*np.ones((NPQ, 1))

fp_Sol_ref = 0.99*np.ones((NPQ, 1))

loadsprofiles  = np.zeros((Nprofiles, nn))
SunIrrprofiles = np.zeros((Nprofiles, nn))

baryLoadold = np.zeros((NPQ, 1), dtype=complex)
suniiold    = np.zeros((NPQ, 1))

cH2       = np.zeros((NPV,nn+1))
cH2old    = 0.0

fpkkold = 0.90

uu       = np.zeros((NPV,nn+1))
uuold    = 0.0
uuoldold = uuold*1
Duu      = 0.0
Duuold   = Duu*1

cant    = np.zeros((Nhid,1))
cantant = np.zeros((Nhid,1))
ekk     = 0

########################################################
#loop through time
for kk in range(nn):
    ##################################
    # instante
    tt = t[kk]
    if np.abs(tt) <= T/2:
        kinit = kk
    print(f'\033[KProgreso: {int(np.round(100*kk/nn))}%            ', end='\r', flush=True)
    ##################################
    # Voltage at regulation nodes
    for node in indexes_reg:
        ve0[node,kk] = ve0_ref[node]    
    ##################################
    # interpole profiles for PQ nodes
    for pp in range(Nprofiles):
        loadsprofiles[pp,kk]  = tm.model_interpole(models[pp], tt)
        SunIrrprofiles[pp,kk] = tm.model_interpole(models[pp+2*Nprofiles], tt)/1000 #sun profiles
        # SunIrrprofiles[pp,kk] = tm.model_interpole(models[pp+Nprofiles], tt)/4/4 #wind profiles
    ##################################
    # Solar generation
    for node in indexes_sol:
        ii = node - NPV
        
        sunii  = installed[node]*random.gauss(1,sigma2)*SunIrrprofiles[0,kk]
        fpSii  = fp_Sol_ref[ii]
        
        P_Sun[ii,kk]  = max(0,sunii)
        Q_Sun[ii,kk]  = max(0,sunii*np.sqrt(fpSii**-2 -1))
    P_Sun_tot = np.sum(P_Sun[:,kk])
               
    ##################################
    # Loads
    for node in indexes_lds:
        ii = node - NPV
        
        #a little bit sun at load nodes
        sunii  = 0.02*installed[node]*random.gauss(1,sigma2)*SunIrrprofiles[0,kk]
        #load interpolation of profiles
        loadii = 0.90*installed[node]*random.gauss(1,sigma2)*profilesatnodes[node,:]@loadsprofiles[:,kk].T
        #power factor interpolation from same profile        
        xfpii  = random.gauss(1,sigma2)*profilesatnodes[node,:]@loadsprofiles[:,kk].T
        fpLii  = min(1, mfp*xfpii + fpLoadmin - mfp*loadfpmin)
        #actual active and reactive load
        pppii  = loadii - sunii
        qqqii  = pppii*np.sqrt(fpLii**-2 -1)
        sssii  = pppii + 1j*qqqii
        
        baryLoad[ii,kk] = np.conj(sssii)#divided by square nominal V=1
    
    ##################################
    # Hydrogen production
    for node in indexes_hid:
        ii            = node - NPV
        Sinstall      = installed[node]
        
        fpSii         = fp_Hyd_ref[ii]
        
        Pnew          = P_Hyd_ref[ii]
        Pnew          = min(Sinstall,Pnew)
        Pnew          = max(0,Pnew)        
        
        P_Hyd[ii,kk]  = Pnew
        Q_Hyd[ii,kk]  = P_Hyd[ii,kk]*np.sqrt(fpSii**-2 - 1)    
    ##################################
    # Define quantities for NR
    barYload = np.diag(baryLoad[:,kk].flatten())
    barS     = np.diag(P_Sun[:,kk]+1j*Q_Sun[:,kk] - P_Hyd[:,kk] - 1j*Q_Hyd[:,kk]  )
    h0       = (np.conj(barY0)@ve0[:,kk]).reshape((NPQ,1))
    #################################
    # NR for voltage in PQ nodes
    vvv      = tm.NRI(barY, h0, barS, barYload, Vinic, itmax, prec)
    ve[:,kk] = vvv*1
    VVV      = np.diag(vvv)
    Vinic    = VVV*1
    #################################
    # Powers depending on voltages
    Sload       = VVV@np.conj(barYload@vvv)
    P_lds[:,kk] = np.real(Sload)
    Q_lds[:,kk] = np.imag(Sload)
    
    V0          = np.diag(ve0[:,kk])
    barS0[:,kk] = V0@np.conj(barY00@ve0[:,kk]) + V0@np.conj(barY0.T@vvv)
    #################################
    # Cost control at regulation nodes
    for ii in range(NPV):
        node            = indexes_reg[ii]
        cH2[node,kk+1]  = cH2old
        uu[node,kk+1]   = uuold
        
        rr0 = (kk-delay0[ii])%ctrlsteps0[ii]
        if rr0 == 0:
            Skk          = barS0[node,kk]
            Pkk          = np.real(Skk)
            
            ekk          = P0ref-Pkk
                        
            uunew      = uuold - 2*ekk
                        
            #communicate as cost
            cH2old         = uunew
            cH2[node,kk+1] = cH2old
            uu[node,kk+1]  = uunew
            
            #update old Preq
            uuoldold      = uuold
            uuold         = uunew
            
    #################################
    # Cost control at H2 nodes
    for ii in range(Nhid):
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        ckk = cH2[0,kk]
        if ckk != cant[ii]:
            node            = indexes_hid[ii] - NPV
            Sinstall        = installed[node]
            
            if ckk<pH2*eta[ii]:
                P_Hyd_ref[node] = Sinstall
            else:
                P_Hyd_ref[node] = 0
                    
            cant[ii]        = ckk

print(' ')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#sunny hours
totalPVgen = np.sum(P_Sun[np.array(indexes_sol)-NPV,:],axis=0)
maxPvgen   = np.max(totalPVgen)
mingenfrac = 0.1/100
sunny      = totalPVgen>maxPvgen*mingenfrac
alfasun    = 0.1

in_region = False
start     = None
sun_range = []

for ii in range(len(sunny)):
    if sunny[ii] and not in_region:
        # Start of a True region
        in_region = True
        start     = t[ii]/60
    elif not sunny[ii] and in_region:
        # End of a True region
        in_region = False
        end       = t[ii]/60
        sun_range.append([start,end])
########################################################
#plot quantities
########################################################
# #profiles
# fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))
# axes[0].plot(t / 60, np.transpose(loadsprofiles))
# axes[0].set_title('Load profiles')

# axes[1].plot(t / 60, np.transpose(SunIrrprofiles))
# axes[1].set_title('Sun profiles')
# for ax in axes.flat:
#     ax.set_xlim(0, t[-1]/60)
    
#     ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
#     ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
#     ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
#     ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
#     ax.set_xticks(custom_ticks)
#     ax.set_xticklabels(custom_labels)
#     ax.set_xlabel('Day Time (hrs)')
    
#     for rr in sun_range:
#         ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
# fig.tight_layout()
########################################################
#LOAD and PV
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*P_lds[np.array(indexes_lds)-NPV,:]))
axes[0].set_title('Power load kW')

axes[1].plot(t / 60, np.transpose(S*P_Sun[np.array(indexes_sol)-NPV,:]))
axes[1].set_title('PV injected power kW')


for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
    
    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()

file_name = 'ex_load_pv'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
#Regulation and H2
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*np.real(barS0)))
axes[0].plot(np.array([t[0],t[-1]]) / 60, S*P0ref*np.array([1,1]))
axes[0].set_title('Regulation injected power kW')

axes[1].plot(t / 60, np.transpose(S*P_Hyd[np.array(indexes_hid)-NPV,:]))
axes[1].set_title('H2 consumed power kW')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
    
    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()

file_name = 'ex_linear_P0_PH2'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
#H2 voltages
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(abs(ve[np.array(indexes_hid)-NPV,:])))
axes[0].set_title('Voltage at H2 nodes')

axes[1].plot(t / 60, np.transpose(np.angle(ve[np.array(indexes_hid)-NPV,:])))
axes[1].set_title('Angle at H2 nodes')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
    
    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()
########################################################
# H2 convex cost
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Convex H2 cost")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#initial conditions for NR iterations on PQ nodes
Vinic = R0 @ expm(1j * Phi0)
########################################################
#prefill vectors and matrices
ve        = np.zeros((NPQ, nn), dtype=complex) #voltage at PQ nodes

barS0     = np.zeros((NPV, nn), dtype=complex)

P_lds     = np.zeros((NPQ, nn))
P_Hyd     = np.zeros((NPQ, nn))
Q_lds     = np.zeros((NPQ, nn))
Q_Hyd     = np.zeros((NPQ, nn))

P_Hyd_ref  = np.zeros((NPQ, 1))
fp_Hyd_ref = 0.99*np.ones((NPQ, 1))

cH2       = np.zeros((NPV,nn+1))
cH2old    = 0.0

uu       = np.zeros((NPV,nn+1))
uuold    = 0.0
uuoldold = uuold*1
Duu      = 0.0
Duuold   = Duu*1

cant    = np.zeros((Nhid,1))
cantant = np.zeros((Nhid,1))
ekk     = 0

E_Hyd     = np.zeros((NPQ, nn+1))
U_Hyd     = np.zeros((NPQ, nn+1))

XXX        = np.zeros((NPQ,NPQ))
aaa        = np.zeros(nn, dtype=complex)
bbb        = np.zeros(nn, dtype=complex)
########################################################
#loop through time
for kk in range(nn):
    ##################################
    # instante
    tt = t[kk]
    if np.abs(tt) <= T/2:
        kinit = kk
    print(f'\033[KProgreso: {int(np.round(100*kk/nn))}%            ', end='\r', flush=True)
    ##################################
    # Hydrogen production
    for node in indexes_hid:
        ii            = node - NPV
        Sinstall      = installed[node]
        
        fpSii         = fp_Hyd_ref[ii]
        
        Pnew          = P_Hyd_ref[ii]
        Pnew          = min(Sinstall,Pnew)
        Pnew          = max(0,Pnew)
        
        XXX[ii,ii]    = np.sqrt(fpSii**-2 - 1)
        
        P_Hyd[ii,kk]  = Pnew
        Q_Hyd[ii,kk]  = P_Hyd[ii,kk]*XXX[ii,ii]
        
        rho            = cH2[0,kk]
        E_Hyd[ii,kk+1] = E_Hyd[ii,kk] + eta[ii]*P_Hyd[ii,kk]*T
        U_Hyd[ii,kk+1] = U_Hyd[ii,kk] + eta[ii]*pH2*S*P_Hyd[ii,kk]*T - rho*((S*P_Hyd[ii,kk]*T)**2)
    ##################################
    # Define quantities for NR
    barYload = np.diag(baryLoad[:,kk].flatten())
    barSant  = barS
    barS     = np.diag(P_Sun[:,kk]+1j*Q_Sun[:,kk] - P_Hyd[:,kk] - 1j*Q_Hyd[:,kk]  )
    h0       = (np.conj(barY0)@ve0[:,kk]).reshape((NPQ,1))
    #################################
    # NR for voltage in PQ nodes
    vvv      = tm.NRI(barY, h0, barS, barYload, Vinic, itmax, prec)
    ve[:,kk] = vvv*1
    VVV      = np.diag(vvv)
    Vinic    = VVV*1
    #################################
    # Powers depending on voltages
    Sload       = VVV@np.conj(barYload@vvv)
    P_lds[:,kk] = np.real(Sload)
    Q_lds[:,kk] = np.imag(Sload)
    
    V0          = np.diag(ve0[:,kk])
    barS0[:,kk] = V0@np.conj(barY00@ve0[:,kk]) + V0@np.conj(barY0.T@vvv)
    #################################
    #matrices for analysis
    AAA = np.conj(diagbarY0) + np.diag(np.conj(barY@vvv))
    BBB = VVV@np.conj(barY)
    
    FFF = np.linalg.inv(BBB)@AAA
    HHH = np.linalg.inv(np.eye(NPQ,NPQ) - FFF@np.conj(FFF) )@np.linalg.inv(BBB)
    
    MMM = np.block( [[-np.conj(FFF)@HHH, np.conj(HHH)],[HHH, -FFF@np.conj(HHH)]]  )
       
    mmm = 0.5*np.ones((1,NPQ))@np.block([diagbarY0,np.conj(diagbarY0)])@MMM
        
    IIIXXX = np.block([[np.eye(NPQ,NPQ) + 1j*XXX],[ np.eye(NPQ,NPQ) - 1j*XXX]])
    
    aaa[kk] = -0.5*pH2*mmm@IIIXXX@longeta/(c*T*S)
    
    dbarS = (barS - barSant)@np.ones((NPQ,1))/T
    if kk==0:
        dbarS = dbarS*0

    bbb[kk] = mmm@np.block([[dbarS],[np.conj(dbarS)]])
    #################################
    # Cost control at regulation nodes
    for ii in range(NPV):
        node            = indexes_reg[ii]
        cH2[node,kk+1]  = cH2old
        uu[node,kk+1]   = uuold
        
        rr0 = (kk-delay0[ii])%ctrlsteps0[ii]
        if rr0 == 0:
            Skk          = barS0[node,kk]
            Pkk          = np.real(Skk)
            
            ekk          = P0ref-Pkk
            De           = ekk - eant
            
            Duu          = uuold-uuoldold
            
            uunew        = uuold + (T0**2)*K*ekk  + Duu + L*T0*De
                        
            #communicate as cost
            cH2old         = max(0.00000000001,c/uunew)
            cH2[node,kk+1] = cH2old
            uu[node,kk+1]  = uunew
            
            #update old Preq
            uuoldold      = uuold
            uuold         = uunew
            Duuold        = Duu
            eant          = ekk
            
    #################################
    # Cost control at H2 nodes
    for ii in range(Nhid):
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        rho = cH2[0,kk]
        if rho != cant[ii]:
            node            = indexes_hid[ii] - NPV
            
            P_Hyd_ref[node] = 0.5*pH2*eta[ii]/(rho*T0*S)
            cant[ii]        = rho

print(' ')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#LMI stability analysis from an estimation of aaa
aaa0    = np.real(aaa[t>0]).mean()
epsilon = 0.001*np.abs(aaa0)

GGG0 = np.block( [[0,1],[-aaa0*K, -aaa0*L]]  )
mmm  = np.block( [[0],[1]]  )
nnn  = np.block( [[-K],[-L]]  )

nP    = GGG0.shape[0]
PPP   = cp.Variable((nP, nP), symmetric=True)
alfa  = cp.Variable((1,1),nonneg=True)
MMM   = cp.bmat([
    [GGG0.T @ PPP + PPP @ GGG0 + alfa*nnn@nnn.T,       epsilon*PPP @ mmm],
    [epsilon*mmm.T @ PPP,           -alfa]
])

constraints = [
    alfa >> 1e-2,   # alfa ≻ 0
    PPP  >> 1e-1 * np.eye(nP),   # P ≻ 0
    MMM  << -1e-6 * np.eye(nP+1) # LMI condition
]
# prob = cp.Problem(cp.Minimize(0), constraints)
prob = cp.Problem(cp.Minimize(cp.trace(PPP)), constraints)

prob.solve(solver=cp.SCS)

print("Status:", prob.status)
print("P =", PPP.value)
print("alfa =", alfa.value)

print("eig(P) =", np.linalg.eigvals(PPP.value) )
print("eig(G0) =", np.linalg.eigvals(GGG0) )

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#plot quantities
########################################################
#Regulation
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*np.real(barS0)))
axes[0].plot(np.array([t[0],t[-1]]) / 60, S*P0ref*np.array([1,1]))
axes[0].set_title('Regulation injected power kW')

axes[1].plot(t / 60, np.transpose(cH2[0,:-1]))
axes[1].set_title('Communication parameter rho')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
    
    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()

file_name = 'ex_convex_P0_rho'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
#H2
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*P_Hyd[np.array(indexes_hid)-NPV,:]))
axes[0].set_title('H2 consumed power kW')

axes[1].plot(t / 60, np.transpose(U_Hyd[np.array(indexes_hid)-NPV,:-1]))
axes[1].set_title('H2 production utility')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
    
    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()

file_name = 'ex_convex_H2'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
# voltages
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(abs(ve[np.array(indexes_hid)-NPV,:])))
axes[0].set_title('Voltage at H2 nodes')

axes[1].plot(t / 60, np.transpose(np.angle(ve[np.array(indexes_hid)-NPV,:])))
axes[1].set_title('Angle at H2 nodes')


for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
    
    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()
########################################################
#matrix analysis
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t[t>0] / 60, np.transpose(np.real(aaa[t>0])))
axes[0].plot(np.array([t[0],t[-1]]) / 60, np.real(aaa0.flatten())*np.array([1,1]), linestyle='--',color='red')
axes[0].plot(np.array([t[0],t[-1]]) / 60, np.real(aaa0+ epsilon).flatten()*np.array([1,1]), linestyle=':',color='red')
axes[0].plot(np.array([t[0],t[-1]]) / 60, np.real(aaa0- epsilon).flatten()*np.array([1,1]), linestyle=':',color='red')
axes[0].set_title('real a')

axes[1].plot(t[t>0][:-1] / 60, np.transpose(np.diff(np.real(aaa[t>0])/T)))
axes[1].set_title('d real a')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')

    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()

file_name = 'matrixanalysis'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t[t>0] / 60, np.transpose(np.real(bbb[t>0])))
axes[0].set_title('real b')

axes[1].plot(t[t>0][:-1] / 60, np.transpose(np.diff(np.real(bbb[t>0])/T)))
axes[1].set_title('real db')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    
    ax.xaxis.set_major_locator(FixedLocator(custom_ticks))
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.75, color='gray')
    
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')

    for rr in sun_range:
        ax.axvspan(rr[0], rr[1], facecolor='#fffff0')
fig.tight_layout()
