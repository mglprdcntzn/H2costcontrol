import numpy as np
import matplotlib.pyplot as plt

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
Nsol          = 20
Nhid          = 10

#choose which nodes are reg, sun, hid, and lds 
indexes_reg, indexes_hid, indexes_sol, indexes_lds = ct.choose_nodes(nodes, lines, NPV, Nsol, Nhid)
#reorder nodes to have NPV first
nodes, lines, indexes_reg,indexes_sol,indexes_hid,indexes_lds = ct.reorder_nodes(nodes, lines, indexes_reg,indexes_sol,indexes_hid,indexes_lds)

Nlds         = len(indexes_lds)
########################################################
#load DG and load profiles
Nprofiles = 3 #number of profiles of each type (load, sun, wind)
models    = tm.built_profiles_models('perfiles.csv')
sigma2    = 0.0001

fpLoadmax = 0.95
fpLoadmin = 0.75
loadfpmin = 0.4
mfp       = (fpLoadmax - fpLoadmin)/(1- loadfpmin)
########################################################
ST = 1200 #[kVA] rate trafos
S  = ST*(N-NPV)
########################################################
#loads at nodes
load, loadmix = ct.load_circuit(Nlds, ST/2, 0.01*ST,Nprofiles) #in kW
pv            = ct.DG_circuit(Nsol, 1*ST/4, 0.01*ST)  #in kW
hid           = ct.DG_circuit(Nhid, 8*ST/4, 0.01*ST)  #in kW

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
t0 =-0.10*24*60  #begining of time in min
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
# MAX H2 PRODUCTION
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Maximum H2 Production")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#Control periods
T0   = 10 #time for Ctrl of Regulation nodes
n0   = np.floor(T0/T)
ndev = 4

ctrlsteps0 = np.random.randint(max(1,n0-ndev),n0+ndev,NPV)
delay0     = np.random.randint(1,n0,NPV)
T0real     = ctrlsteps0.reshape((NPV,1))*T

T1   = 2 #time for Ctrl of H2 nodes
n1   = np.floor(T1/T)
ndev = 4

ctrlsteps1 = np.random.randint(max(1,n1-ndev),n1+ndev,Nhid)
delay1     = np.random.randint(1,n1,Nhid)
T1real     = ctrlsteps1.reshape((Nhid,1))*T
########################################################
#Regulation control gains.
pH2  = 5

P0ref  = 1.00*np.sum(installed[indexes_lds])
fp0min = 0.93

eant    = np.zeros((NPV,1))
eantant = np.zeros((NPV,1))

K = 0.00001
L = 0.005

########################################################
#H2 Control gains
eta     = 0.50 + 0.50*np.random.rand(Nhid)

longeta = np.zeros((N,1))
longeta[indexes_hid] = eta.reshape(Nhid,1)
longeta = longeta[NPV:]

cant    = 10*np.ones((Nhid,1))
cantant = 10*np.ones((Nhid,1))
ekk     = 0

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

P_Hyd_ref  = np.zeros((NPQ, 1))
fp_Hyd_ref = 0.99*np.ones((NPQ, 1))

fp_Sol_ref = 0.99*np.ones((NPQ, 1))

baryLoad   = np.zeros((NPQ, nn), dtype=complex)

vePV_mes  = expm(1j*0.0005*np.diag(np.random.randn(NPV)))
vePVinic  = np.diag(0.00005*np.random.randn(NPV)+1)@vePV_mes@np.ones((NPV, 1), dtype=complex)
vePVctrl  = np.ones((NPV, 1), dtype=complex)

loadsprofiles  = np.zeros((Nprofiles, nn))
SunIrrprofiles = np.zeros((Nprofiles, nn))

barS0     = np.zeros((NPV, nn), dtype=complex)

P_Hyd     = np.zeros((NPQ, nn))
Q_Hyd     = np.zeros((NPQ, nn))

E_Hyd     = np.zeros((NPQ, nn+1))
U_Hyd     = np.zeros((NPQ, nn+1))

P_Hyd_ref  = 0*(0.6*ST/S)*np.ones((NPQ, 1))
fp_Hyd_ref = 0.99*np.ones((NPQ, 1))

fp_Sol_ref = 0.99*np.ones((NPQ, 1))

vePV_mes  = expm(1j*0.0005*np.diag(np.random.randn(NPV)))
vePVinic  = np.diag(0.00005*np.random.randn(NPV)+1)@vePV_mes@np.ones((NPV, 1), dtype=complex)
vePVctrl  = np.ones((NPV, 1), dtype=complex)

loadsprofiles  = np.zeros((Nprofiles, nn))
SunIrrprofiles = np.zeros((Nprofiles, nn))


baryLoadold = np.zeros((NPQ, 1), dtype=complex)
suniiold    = np.zeros((NPQ, 1))

cost    = (1/0.004)*np.ones((NPV,nn+1))
cold    = (1/0.004)
coldold = cold
Dcost   = 0.0

fpkkold = 0.90

Preq       = 0.004*np.ones((NPV,nn+1))
Preqold    = 0.004
Preqoldold = Preqold
DPreq      = 0.0
DPreqold   = DPreq

XXX        = np.zeros((NPQ,NPQ))
aaa        = np.zeros(nn, dtype=complex)
bbb        = np.zeros(nn, dtype=complex)
eigMMM     = np.zeros((2,nn), dtype=complex)

vvv = 0.996*expm(1j*-0.003)*np.ones((NPQ,1))
AAA = np.conj(diagbarY0) + np.diag(np.conj(barY@vvv).flatten())
BBB = np.conj(barY)@np.diag(np.conj(vvv).flatten())

FFF = np.linalg.inv(BBB)@AAA
HHH = np.linalg.inv(np.eye(NPQ,NPQ) - FFF@np.conj(FFF) )@np.linalg.inv(BBB)

MMM = np.block( [[-np.conj(FFF)@HHH, np.conj(HHH)],[HHH, -FFF@np.conj(HHH)]]  )

bbb0 = 0.5*np.ones((1,NPQ))@np.block([diagbarY0,np.conj(diagbarY0)])@MMM

IIIXXX = np.block([[np.eye(NPQ,NPQ) + 1j*XXX],[ np.eye(NPQ,NPQ) - 1j*XXX]])

aaa0 = -np.real(0.5*pH2*bbb0@IIIXXX@longeta)

barS = np.zeros((NPQ,1))

########################################################
#initial conditions for NR iterations on PQ nodes
R0    = np.eye(NPQ)
Phi0  = np.zeros((NPQ, NPQ))
Vinic = R0 @ expm(1j * Phi0)
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
        
        E_Hyd[ii,kk+1] = E_Hyd[ii,kk] + eta[ii]*P_Hyd[ii,kk]*T
        U_Hyd[ii,kk+1] = U_Hyd[ii,kk] + eta[ii]*pH2*P_Hyd[ii,kk]*T - cost[0,kk]*((P_Hyd[ii,kk]*T)**2)
       
    ##################################
    # Loads
    for node in indexes_lds:
        ii = node - NPV
        
        #a little bit sun at load nodes
        sunii  = 0.10*installed[node]*random.gauss(1,sigma2)*SunIrrprofiles[0,kk]
        #load interpolation of profiles
        loadii = installed[node]*random.gauss(1,sigma2)*profilesatnodes[node,:]@loadsprofiles[:,kk].T
        #power factor interpolation from same profile        
        xfpii  = random.gauss(1,sigma2)*profilesatnodes[node,:]@loadsprofiles[:,kk].T
        fpLii  = min(1, mfp*xfpii + fpLoadmin - mfp*loadfpmin)
        #actual active and reactive load
        pppii  = loadii - sunii
        qqqii  = pppii*np.sqrt(fpLii**-2 -1)
        sssii  = pppii + 1j*qqqii
        
        baryLoad[ii,kk] = np.conj(sssii)#divided by square nominal V=1
        
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
    # barS0[:,kk] = 
    #################################
    #matrices for analysis
    AAA = np.conj(diagbarY0) + np.diag(np.conj(barY@vvv))
    BBB = VVV@np.conj(barY)
    
    FFF = np.linalg.inv(BBB)@AAA
    HHH = np.linalg.inv(np.eye(NPQ,NPQ) - FFF@np.conj(FFF) )@np.linalg.inv(BBB)
    
    MMM = np.block( [[-np.conj(FFF)@HHH, np.conj(HHH)],[HHH, -FFF@np.conj(HHH)]]  )
       
    mmm = 0.5*np.ones((1,NPQ))@np.block([diagbarY0,np.conj(diagbarY0)])@MMM
        
    IIIXXX = np.block([[np.eye(NPQ,NPQ) + 1j*XXX],[ np.eye(NPQ,NPQ) - 1j*XXX]])
    
    aaa[kk] = -0.5*pH2*mmm@IIIXXX@longeta
    
    dbarS = (barS - barSant)@np.ones((NPQ,1))/T
    if kk==0:
        dbarS = dbarS*0

    bbb[kk] = mmm@np.block([[dbarS],[np.conj(dbarS)]])
    
    MMM = np.array([[0, 1],[-aaa[kk]*K, -aaa[kk]*L]])
    eigMMM[:,kk] = np.linalg.eigvals(MMM)
    
    
    #################################
    # Cost control at regulation nodes
    for ii in range(NPV):
        node            = indexes_reg[ii]
        cost[node,kk+1] = cold
        Preq[node,kk+1] = Preqold
        
        rr0 = (kk-delay0[ii])%ctrlsteps0[ii]
        if rr0 == 0:
            Skk          = barS0[node,kk]
            Pkk          = np.real(Skk)
            Qkk          = np.imag(Skk)
            fpkk         = np.abs(Pkk/Skk)
            
            # if kk<nn/2:
            #     ekk          = P0ref-Pkk
            # else:
            #     ekk          = 1.1*P0ref-Pkk
            ekk          = P0ref-Pkk
            De           = ekk - eant
            
            DPreq        = Preqold-Preqoldold
            Dfp          = fpkk - fpkkold
            
            Preqnew      = Preqold + (T0**2)*K*ekk  + DPreq + L*T0*De
            
            # Preqnew      = max(Preqnew, 1/S)
            
            #communicate as cost
            cold            = 1/Preqnew
            cost[node,kk+1] = cold
            Preq[node,kk+1] = Preqnew
            
            #update old Preq
            Preqoldold      = Preqold
            Preqold         = Preqnew
            DPreqold        = DPreq
            fpkkold         = fpkk
            eant            = ekk
            
            

    #################################
    # Cost control at H2 nodes
    for ii in range(Nhid):
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        # if kk>delay1[ii]:
        #     ckk = cost[0,kk-delay1[ii]]
        # else:
        ckk = cost[0,kk]
        if ckk != cant[ii]:
        # if rr1 == 0:
            node            = indexes_hid[ii] - NPV
            Sinstall        = installed[node]
            
            # if ckk != cant[ii]:
            #     cant[ii] = ckk
            #     if ckk<pH2*eta[ii]:
            #         P_Hyd_ref[node] = Sinstall
            #     else:
            #         P_Hyd_ref[node] = 0
            
            P_Hyd_ref[node] = 0.5*pH2*eta[ii]/ckk
            
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

# for ax in axes.flat:
#     ax.set_xlim(0, t[-1]/60)
#     ax.grid(True)
#     ax.set_xticks(custom_ticks)
#     ax.set_xticklabels(custom_labels)
#     ax.set_xlabel('Day Time (hrs)')
# fig.tight_layout()        

########################################################
#plot quantities
########################################################
#LOAD
fig, axes = plt.subplots(2,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0][0].plot(t / 60, np.transpose(S*P_lds[np.array(indexes_lds)-NPV,:]))
axes[0][0].set_title('Power load MW')

fp_lds = np.abs(P_lds[np.array(indexes_lds)-NPV,:]/(P_lds[np.array(indexes_lds)-NPV,:] + 1j*Q_lds[np.array(indexes_lds)-NPV,:]))

axes[0][1].plot(t / 60, np.transpose(fp_lds))
axes[0][1].set_title('Load power factor')


axes[1][0].plot(t / 60, np.transpose(S*np.sum(P_lds[np.array(indexes_lds)-NPV,:],axis=0)))
axes[1][0].set_title('Total Power load MW')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()
########################################################
#PV
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*P_Sun[np.array(indexes_sol)-NPV,:]))
axes[0].set_title('PV injected power MW')

fp_sun = np.abs(P_Sun[np.array(indexes_sol)-NPV,:]/(P_Sun[np.array(indexes_sol)-NPV,:] + 1j*Q_Sun[np.array(indexes_sol)-NPV,:]))

axes[1].plot(t / 60, np.transpose(fp_sun))
axes[1].set_title('PV injection power factor')
# axes[1].plot(t[:-1] / 60, np.transpose(np.diff(S*P_Sun[np.array(indexes_sol)-NPV,:])))
# axes[1].set_title('PV injection derivative')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()
########################################################
#Regulation
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*np.real(barS0)))
axes[0].plot(np.array([t[0],t[-1]]) / 60, S*P0ref*np.array([1,1]))
axes[0].set_title('Regulation injected power MW')

fp_0 = np.abs(np.real(barS0)/barS0)

axes[1].plot(t / 60, np.transpose(fp_0))
axes[1].set_title('Regulation injection power factor')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()
########################################################
#Regulation requirements and cost
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*Preq[:,:-1]))
axes[0].set_title('Required power by regulator')

axes[1].plot(t / 60, np.transpose(cost[:,:-1]))
axes[1].set_title('Energy cost imposed by regulator')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()
########################################################
#H2
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*P_Hyd[np.array(indexes_hid)-NPV,:]))
axes[0].set_title('H2 consumed power MW')

# fp_H2 = np.abs(P_Hyd[np.array(indexes_hid)-NPV,:]/(P_Hyd[np.array(indexes_hid)-NPV,:] + 1j*Q_Hyd[np.array(indexes_hid)-NPV,:]))

# axes[2].plot(t / 60, np.transpose(fp_H2))
# axes[2].set_title('H2 consumption power factor')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()

#H2 with respect to market price
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*P_Hyd[np.array(indexes_hid)-NPV,:]/hid[:,None])) #hid[:,None]
axes[0].set_title('H2 consumed power MW per installed capacity')

axes[1].plot(t / 60, np.transpose(U_Hyd[np.array(indexes_hid)-NPV,:-1]/pH2))
axes[1].set_title('H2 production utility per installed market price')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()

#H2 sum with respect to requiered
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0].plot(t / 60, np.transpose(S*np.sum(P_Hyd[np.array(indexes_hid)-NPV,:],axis=0)))
axes[0].plot(t / 60, np.transpose(S*Preq[:,:-1]))
axes[0].set_title('Total H2 consumed power MW')

axes[1].plot(t / 60, np.transpose(np.sum(P_Hyd[np.array(indexes_hid)-NPV,:],axis=0)/Preq[:,:-1]))
axes[1].set_title('Total H2 consumed power w/r to required power')



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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()
########################################################
#matrix analysis
fig, axes = plt.subplots(3,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0][0].plot(t[t>0] / 60, np.transpose(np.real(aaa[t>0])))
axes[0][0].plot(np.array([t[0],t[-1]]) / 60, np.real(aaa0.flatten())*np.array([1,1]), linestyle='--')
axes[0][0].set_title('real a')

axes[0][1].plot(t[t>0][:-1] / 60, np.transpose(np.diff(np.real(aaa[t>0])/T)))
axes[0][1].set_title('d real a')

axes[1][0].plot(t[t>0] / 60, np.transpose(np.real(bbb[t>0])))
axes[1][0].set_title('real b')

axes[1][1].plot(t[t>0][:-1] / 60, np.transpose(np.diff(np.real(bbb[t>0])/T)))
axes[1][1].set_title('real db')

axes[2][0].plot(t[t>0] / 60, np.transpose(np.real(eigMMM[:,t>0])))
axes[2][0].set_title('real eig MMM')

axes[2][1].plot(t[t>0] / 60, np.transpose(np.imag(eigMMM[:,t>0])))
axes[2][1].set_title('imag eig MMM')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()

########################################################
fig, axes = plt.subplots(4,2,figsize=(15 * 2 / 2.54, 10 * 2 / 2.54))

axes[0][0].plot(t / 60, np.transpose(abs(ve0)))
axes[0][0].set_title('Voltage at regulation nodes')

axes[0][1].plot(t / 60, np.transpose(np.angle(ve0)))
axes[0][1].set_title('Angle at regulation nodes')

axes[1][0].plot(t / 60, np.transpose(abs(ve[np.array(indexes_sol)-NPV,:])))
axes[1][0].set_title('Voltage at PV nodes')

axes[1][1].plot(t / 60, np.transpose(np.angle(ve[np.array(indexes_sol)-NPV,:])))
axes[1][1].set_title('Angle at PV nodes')

axes[2][0].plot(t / 60, np.transpose(abs(ve[np.array(indexes_hid)-NPV,:])))
axes[2][0].set_title('Voltage at H2 nodes')

axes[2][1].plot(t / 60, np.transpose(np.angle(ve[np.array(indexes_hid)-NPV,:])))
axes[2][1].set_title('Angle at H2 nodes')

axes[3][0].plot(t / 60, np.transpose(abs(ve[np.array(indexes_lds)-NPV,:])))
axes[3][0].set_title('Voltage at load nodes')

axes[3][1].plot(t / 60, np.transpose(np.angle(ve[np.array(indexes_lds)-NPV,:])))
axes[3][1].set_title('Angle at load nodes')

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
        ax.axvspan(rr[0], rr[1], facecolor='yellow', alpha=alfasun)
fig.tight_layout()