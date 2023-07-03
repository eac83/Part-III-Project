import numpy as np
import h5py as hp
import pandas as pd
from astropy import constants as const

class Halo():
	def __init__(self, pos, m200c, r200c):
		self.pos = pos
		self.m200c = m200c
		self.r200c = r200c

	def nearest(self, coord, boxSize):
    for axis in range(3):
        ndx = np.where((coord[:,axis] > boxSize/2))
        coord[ndx,axis] = boxSize - coord[ndx,axis]
        ndx = np.where((coord[:,axis] < -boxSize/2))
        coord[ndx,axis] = boxSize + coord[ndx,axis]
    return coord

def ZoomDensities(directories, redshift, xBins):
    #set up return arrays
    DensityProfile = np.empty([xBins.shape[0]-1])                                           #10^10 Msun/h (cMpc/h)^-3
    MassHalo = np.array([])                                                                 #10^10 Msun/h
    R_200 = np.array([])                                                                    #cMpc/h

    for directory in directories:
        #find correct snapshot
        for snapshot in range(0,700):
            with hp.File(f'{directory}/snap_{str(snapshot).zfill(3)}.hdf5') as snap:
                z = snap['Header'].attrs['Redshift']
                if abs(z-redshift)<=0.01:
                    break
        print(f'{directory}, {snapshot}, z={z}')

        #grab data
        with hp.File(f'{directory}/snap_{str(snapshot).zfill(3)}.hdf5') as snap, hp.File(f'{directory}/fof_subhalo_tab_{str(snapshot).zfill(3)}.hdf5') as fof:
            boxSize = snap['Header'].attrs['BoxSize']                                       #cMpc/h

            #halo data
            fofMassHalo = np.array(fof.get('Group/Group_M_Crit200'))
            idx = np.where(fofMassHalo == np.amax(fofMassHalo))[0]
            MassHalo = np.append(MassHalo, fofMassHalo[idx])                                #10^10 Msun/h
            CoMHalo = np.array(fof.get('Group/GroupPos'))[idx][0]
            r200 = np.array(fof.get('Group/Group_R_Crit200'))[idx][0]                               #cMpc/h
            R_200 = np.append(R_200, np.array(fof.get('Group/Group_R_Crit200'))[idx])       #cMpc/h
            #particle data
            CoordsGas = np.array(snap.get('PartType0/Coordinates'))                         #cMpc/h
            CoordsDM = np.array(snap.get('PartType1/Coordinates'))                          #cMpc/h
            CoordsStars = np.array(snap.get('PartType4/Coordinates'))                       #cMpc/h
            MassGas = np.array(snap.get('PartType0/Masses'))                                #10^10 Msun/h
            MassDM = np.full((CoordsDM.shape[0]), snap['Header'].attrs['MassTable'][1])     #10^10 Msun/h
            MassStars = np.array(snap.get('PartType4/Masses'))                              #10^10 Msun/h

        MassProfile_i = np.empty([xBins.shape[0]-1])                                        #10^10 Msun/h

        #move into halo FoR
        X_Gas_Halo = nearest(CoordsGas - CoMHalo, boxSize)                                  #cMpc/h
        X_DM_Halo = nearest(CoordsDM - CoMHalo, boxSize)                                    #cMpc/h
        X_Stars_Halo = nearest(CoordsStars - CoMHalo, boxSize)                              #cMpc/h
        D_Gas_Halo = np.sqrt(np.sum(X_Gas_Halo**2, axis=1))#/R_200[-1]                       #1
        D_DM_Halo = np.sqrt(np.sum(X_DM_Halo**2, axis=1))#/R_200[-1]                         #1 
        D_Stars_Halo = np.sqrt(np.sum(X_Stars_Halo**2, axis=1))#/R_200[-1]                   #1  
        r = xBins*r200
        #add mass of particles within spherical shells
        for j in range(0, len(xBins)-1):
            massGas = MassGas[np.where((D_Gas_Halo>r[j])&(D_Gas_Halo<r[j+1]))].sum()
            massDM = np.sum(MassDM[np.where((D_DM_Halo>=r[j])&(D_DM_Halo<r[j+1]))])
            massStars = np.sum(MassStars[np.where((D_Stars_Halo>=r[j])&(D_Stars_Halo<r[j+1]))])
            #MassProfile_i[j] = MassGas[np.where((D_Gas_Halo > xBins[j]) & (D_Gas_Halo < xBins[j+1]))].sum()+MassDM[np.where((D_DM_Halo > xBins[j]) & (D_DM_Halo < xBins[j+1]))].sum()+MassStars[np.where((D_Stars_Halo > xBins[j]) & (D_Stars_Halo < xBins[j+1]))].sum()
            MassProfile_i[j] = massGas+massDM+massStars
        #divide by volume of shell
        DensityProfile_i = np.divide(MassProfile_i,(4/3*np.pi*((r[1:])**3-(r[:-1])**3)))
        DensityProfile = np.vstack((DensityProfile.T, DensityProfile_i)).T
    return DensityProfile[:,1:], MassHalo, R_200

def MassAveragedProfiles(DensityProfiles, MassHalo, xBins, mBins):
    DensityAvg = np.empty([xBins.shape[0]-1])
    n = np.array([])                        #number of haloes in that bin

    for i, m_i in enumerate(mBins[:-1]):
        idx = np.array(np.where((MassHalo*1e10 >= mBins[i])&(MassHalo*1e10 < mBins[i+1])))[0]
        n = np.append(n, idx.shape[0])
        if n[-1] > 0:
            DensityAvg = np.vstack((DensityAvg,((np.sum(MassHalo[idx]*DensityProfiles[:,idx],axis=1)))/np.sum(MassHalo[idx])))
        else:
            DensityAvg = np.vstack((DensityAvg,np.zeros(xBins.shape[0]-1)))    #if no haloes in bin, just set <\rho>=0\forall r  
            
    return DensityAvg[1:,:], n

def u(DensityAvg, MassHalo, R_200, xBins, mBins, kBins):
    invR_200_avg = np.empty([mBins.shape[0]-1])                         #(cMpc/h)^-1
    u = np.empty([mBins.shape[0]-1])                                    #(cMpc/h)^-3
    mMean = np.empty([mBins.shape[0]-1])
    #calculate <1/r_200>
    for i in range(0, len(mBins)-1):
        idx = np.array(np.where((MassHalo*1e10 >= mBins[i])&(MassHalo*1e10 < mBins[i+1])))[0]
        invR_200_avg[i] = np.divide((MassHalo[idx]/R_200[idx]).sum(),MassHalo[idx].sum())
        mMean[i] = (MassHalo[idx].sum())/idx.shape[0]
    xBar = np.sqrt(xBins[1:]*xBins[:-1])                                #1
    rBar = xBar/invR_200_avg[:,np.newaxis]                              #cMpc/h
    #calculate u(k)
    for i, K in enumerate(kBins):
        u = np.vstack((u,(np.trapz(DensityAvg*(rBar)**2*(np.sin(K*rBar)/(K*rBar)), x=rBar)/np.trapz(rBar**2*DensityAvg,x=rBar)).T))
    return u[1:]

def Power1h(U, directory, snapshot, kBins, mBins):
    mBar = np.log10(np.sqrt(mBins[:-1]*mBins[1:]))
    #calculate rho bar
    with hp.File(f'{directory}/snap_{str(snapshot).zfill(3)}.hdf5') as snap:
        Omega0 = snap['Header'].attrs['Omega0']
        h = snap['Header'].attrs['HubbleParam']
        #h=0.679
        z = round(snap['Header'].attrs['Redshift'])

    H = 100*h*1e3#*np.sqrt(Omega0*(1+z)**3+(1-Omega0))                           #m s^-1 Mpc^-1
    rhoCrit = 3*H**2/(8*np.pi*const.G)                                          #kg Mpc^-2 m^-1 
    rhoCrit = (rhoCrit * (const.pc*1e6) / (const.M_sun) / h**2).value           #M_sun/h (cMpc/h)^-3     
    rhoBar = rhoCrit*Omega0                                            #M_sun/h (cMpc/h)^-3
    
    #read in FABLE box halo mass number density, n(m)=dN/dlogm
    df = pd.read_csv(f'./Halo_Mass_Functions/HMF_{float(np.log10(mBins[0]))}-{np.log10(mBins[-1])}-{mBins.shape[0]}_z{z}.csv')
    n = np.array(df.iloc[:,1])
    #print(n)                                                  #(cMpc/h)^-3
    mMean = np.array(df.iloc[:,0])                                              #M_sun/h 
    P = np.array([])                                                            #(cMpc/h)^3
    #mBar = np.sqrt(mBins[1:]*mBins[:-1])                                        #M_sun/h
    for i, K in enumerate(kBins):
        P = np.append(P, np.trapz(n*(mMean/rhoBar)**2*np.abs(U[i,:])**2, x=mBar))
        #P = np.append(P, np.trapz(n*(mMean/rhoBar)**2, x=mBar))
    return P

def AnalyticalP1h(a, b, xBins, mBins, z, binNo):
    xBar = np.sqrt(xBins[:-1]*xBins[1:])
    dfBox = pd.read_csv(f'./Density_Profiles/Density_FABLE_Box_z{z}.csv')
    massBox = np.array(dfBox.iloc[:,1])
    r200Box = np.array(dfBox.iloc[:,-1])
    dfAvgBox = pd.read_csv(f'./Mass_Averaged_Density_Profiles/FABLE_Box_z{z}_{np.log10(mBins[0])}-{np.log10(mBins[-1])}-{mBins.shape[0]}.csv')
    densAvgBox = np.array(dfAvgBox.iloc[:,2:-1])
    densAvgTra = densAvgBox.copy()
    MassA = np.array([])
    MassBox = np.trapz(densAvgBox[binNo,:]*xBar**2,xBar)
    i = 0
    for i, A in enumerate(xBar):
        densAvgTra = densAvgBox.copy()
        densAvgTra[binNo,:i] *= A**-a*xBar[:i]**a
        densAvgTra[binNo,i:] *= A**-b*xBar[i:]**b
        MassA = np.append(MassA,np.trapz(densAvgTra[binNo,:]*xBar**2,xBar))
    i = np.array(np.where(abs(1-MassA/MassBox)==np.amin(abs(1-MassA/MassBox))))[0][0]
    A = xBar[i]
    densAvgTra = densAvgBox.copy()
    densAvgTra[binNo,:i] *= A**-a*xBar[:i]**a
    densAvgTra[binNo,i:] *= A**-b*xBar[i:]**b
    return densAvgTra, massBox, r200Box

def PivotP1h(xBins, mBins, z, i, binNo):
    xBar = np.sqrt(xBins[:-1]*xBins[1:])
    dfBox = pd.read_csv(f'./Density_Profiles/Density_FABLE_Box_z{z}.csv')
    massBox = np.array(dfBox.iloc[:,1])
    r200Box = np.array(dfBox.iloc[:,-1])
    dfAvgBox = pd.read_csv(f'./Mass_Averaged_Density_Profiles/FABLE_Box_z{z}_{np.log10(mBins[0])}-{np.log10(mBins[-1])}-{mBins.shape[0]}.csv')
    densAvgBox = np.array(dfAvgBox.iloc[:,2:-1])
    MassBox = np.trapz(densAvgBox[binNo,:]*xBar**2,xBar)
    Mcalc = np.array([])
    Model = np.array([])
    C = [I*0.01 for I in range(1,200)]
    for c in C:
        D = [I*0.01 for I in range(0,200)]
        for d in D:
            a = xBar[i]
            densAvgTra = densAvgBox.copy()
            densAvgTra[binNo,:i] *= a**-c*xBar[:i]**c
            densAvgTra[binNo,i:] *= a**-d*xBar[i:]**d
            Mcalc = np.append(Mcalc,np.trapz(xBar**2*densAvgTra[binNo,:],xBar))
            Model = np.append(Model,[c,d])
    M = np.trapz(densAvgBox[binNo,:].T*xBar**2, xBar)
    j = np.array(np.where(abs(1-Mcalc/MassBox)==np.amin(abs(1-Mcalc/MassBox))))[0]
    c, d = Model[j*2], Model[j*2+1]
    #print(f'c={c}, d={d}')
    #print(M)
    #print(Mcalc[j])
    densAvgTra = densAvgBox.copy()
    densAvgTra[binNo,:i] *= a**-c*xBar[:i]**c
    densAvgTra[binNo,i:] *= a**-d*xBar[i:]**d

    return densAvgTra, massBox, r200Box

#def haloMassFun(snap, mBins):
 #   with hp.File(f'/data/ERCblackholes4/FABLE/boxes/L40_512_MDRIntRadioEff/fof_subhalo_tab_{str(snap).zfill(3)}.hdf5') as fof:
        
