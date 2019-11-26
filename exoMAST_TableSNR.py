
############################################
# Written by Kevin B Stevenson, March 2019
# For questions email kbstevenson@gmail.com
############################################

import numpy as np

# constants
c    = 2.99792458e8     # m/s, speed of light
h    = 6.6260693e-34    # J s, Planck's constant
k    = 1.3806503e-23    # J/K, Boltzmann's constant
R    = 8.3144598        # J/mol/K, ideal gas constant
G    = 6.67408e-11      # m^3/kg/s^2, Gravitational constant
Rsun = 695508.e3        # m, Radius of Sun
Rjup = 71492.e3         # m, Radius of Jupiter
Mjup = 1.8986e27        # kg, Mass of Jupiter
Re   = 6.3781e6         # m, Radius of Earth
Me   = 5.972e24         # kg, Mass of Earth
au   = 1.4959787066e11  # m, 1 AU
d2s  = 86400.           # Days to seconds

def planetTeq(Ts, aRs, A=0, f=0.5):
    '''
    Compute the planet's equilibrium temperature.
    
    Parameters
    ----------
    Ts      : float or array_like
        Stellar temperature (K)
    aRs     : float or array_like
        Semi-major axis / stellar radius (unitless)
    A       : float
        Albedo (unitless), default is zero
    f       : float
        Heat redistribution factor (unitless), default is 0.5
    
    Returns
    -------
    Tp      : float or array_like
        Planet equilibrium temperature
    '''
    return Ts * (1./aRs)**0.5 * ((1.-A)/(4. * f))**0.25

def planetGravity(Mp, Rp):
    '''
    Compute the planet's surface gravity in mks units.
    
    Parameters
    ----------
    Mp      : float or array_like
        Planet mass (Jupiter mass)
    Rp      : float or array_like
        Planet radius (Jupiter radii)
    
    Returns
    -------
    gp      : float or array_like
        Planet gravity (m/s^2)
    '''
    return G*Mp*Mjup/(Rp*Rjup)**2

def planetLogg(Mp, Rp):
    '''
    Compute the planet's log surface gravity in cgs units.
    
    Parameters
    ----------
    Mp      : float or array_like
        Planet mass (Jupiter mass)
    Rp      : float or array_like
        Planet radius (Jupiter radii)
    
    Returns
    -------
    logg    : float or array_like
        Planet log(gravity) (dex)
    '''
    return np.log10(planetGravity(Mp, Rp)*100)

def planetScaleHeight(Tp, gp, mu=2.3):
    '''
    Compute the planet's atmospheric scale height, H.
    
    Parameters
    ----------
    Tp      : float or array_like
        Planet temperature (K)
    gp      : float or array_like
        Planet gravity (m/s^2)
    mu      : float
        Mean molecular weight (g/mol), default is hydrogen dominated
    
    Returns
    -------
    H       : float or array_like
        Scale height (meters)
    '''
    return 1e3*R*Tp/mu/gp

def planetSignalSize(H, Rp, Rs):
    '''
    Compute the planet's transmission signal size in ppm.
    
    Parameters
    ----------
    H       : float or array_like
        Scale height (meters)
    Rp      : float or array_like
        Planet radius (Jupiter radii)
    Rs      : float or array_like
        Stellar radius (Solar radii)
    
    Returns
    -------
    signal  : float or array_like
        Signal size (ppm)
    '''
    return 2e6*H*Rp*Rjup/(Rs*Rsun)**2

def planck(wave, temp):
    '''
    Calculate bolometric flux, B, at a given wavelength.
    
    Parameters
    ----------
    wave    : float or array_like
        Wavelength (meters)
    temp    : float or array_like
        Temperature (K)
    
    Returns
    -------
    B       : float or array_like
        bolometric flux (W/sr/m^3)
    '''
    c1 = 2. * h * c**2
    c2 = h * c / k
    val =  c2 / wave / temp
    return c1 / ( wave**5 * (np.exp(val) - 1.) )

def planetStarEmission(RpRs, Bp, Bs):
    '''
    Calculate the thermal emission planet-to-star flux ratio, Fp/Fs, in ppm.
    
    Parameters
    ----------
    RpRs    : float or array_like
        Planet-to-star radius ratio
    Bp      : float or array_like
        Planet bolometric flux (W/sr/m^3)
    Bs      : float or array_like
        Stellar bolometric flux (W/sr/m^3)
    
    Returns
    -------
    FpFs    : float or array_like
        Planet-to-star flux ratio (ppm)
    '''
    return 1e6 * (RpRs)**2 * Bp / Bs

def planetStarReflection(RpRs, aRs, Ag):
    '''
    Calculate the reflected light planet-to-star flux ratio in ppm.
    
    Parameters
    ----------
    RpRs    : float or array_like
        Planet-to-star radius ratio
    aRs     : float or array_like
        Semi-major axis / stellar radius (unitless)
    Ag      : float
        Geometric albedo (unitless)
    
    Returns
    -------
    FpFs    : float or array_like
        Planet-to-star flux ratio (ppm)
    '''
    return 1e6*Ag*(RpRs/aRs)**2

def eclipseSNR(RpRs, dur, mag, wave, Tp, Ts, ref_FpFs, ref_dur, ref_mag):
    '''
    Compute the planet's emission signal-to-noise at the given wavelength.
    
    Parameters
    ----------
    RpRs        : float or array_like
        Planet-to-star radius ratio
    dur         : float or array_like
        Eclipse duration (days)
    mag         : float or array_like
        Stellar magnitude
    wave        : float or array_like
        Wavelength (microns)
    Tp          : float or array_like
        Planet temperature (K)
    Ts          : float or array_like
        Stellar temperature (K)
    ref_FpFs    : float or array_like
        Planet-to-star flux ratio of reference system (ppm)
    ref_dur     : float or array_like
        Eclipse duration of reference system (days)
    ref_mag     : float or array_like
        Stellar magnitude of reference system
    
    Returns
    -------
    snr         : float or array_like
        Eclipse signal-to-noise relative to reference system
    '''
    Bp          = planck(wave*1e-6, Tp)
    Bs          = planck(wave*1e-6, Ts)
    FpFs        = planetStarEmission(RpRs, Bp, Bs)
    fluxRatio   = 10**(-0.4*(mag-ref_mag))
    return FpFs/ref_FpFs * np.sqrt(fluxRatio) * np.sqrt(dur/ref_dur)

def transitSNR(signal, dur, mag, ref_signal, ref_dur, ref_mag):
    '''
    Compute the planet's transmission signal-to-noise in the specified band.
    
    Parameters
    ----------
    signal      : float or array_like
        Signal size (ppm)
    dur         : float or array_like
        Transit duration (days)
    mag         : float or array_like
        Stellar magnitude
    ref_signal  : float or array_like
        Signal size of reference system (ppm)
    ref_dur     : float or array_like
        Transit duration of reference system (days)
    ref_mag     : float or array_like
        Stellar magnitude of reference system
    
    Returns
    -------
    snr         : float or array_like
        Transit signal-to-noise relative to reference system
    '''
    fluxRatio   = 10**(-0.4*(mag-ref_mag))
    return signal/ref_signal * np.sqrt(fluxRatio) * np.sqrt(dur/ref_dur)

def TSM(Rp, Mp, Rs, Ts, aRs, jmag):
    '''
    Calculate the transmission spectroscopy metric (TSM) from Kempton et al (2018).
    
    Parameters
    ----------
    Rp          : float or array_like
        Planet radius (Jupiter radii)
    Mp          : float or array_like
        Planet mass (Jupiter mass)
    Rs          : float or array_like
        Stellar radius (Solar radii)
    Ts          : float or array_like
        Stellar temperature (K)
    aRs         : float or array_like
        Semi-major axis / stellar radius (unitless)
    jmag        : float or array_like
        Stellar J-band magnitude
    
    Returns
    -------
    tsm         : float or array_like
        Transmission spectroscopy metric
    '''
    # Planet radius in Earth radii
    Rpe = (Rp*Rjup/Re)  
    # Define radius-dependent scale factor
    if Rpe < 1.5:
        sf = 0.190
    elif Rpe < 2.75:
        sf = 1.26
    elif Rpe < 4.0:
        sf = 1.28
    else:
        sf = 1.15
    Teq = planetTeq(Ts, aRs, A=0, f=1)
    return sf * Rpe**3 * Teq / (Mp*Mjup/Me) / Rs**2 * 10**(-jmag/5.)

def ESM(RpRs, Ts, aRs, kmag):
    '''
    Calculate the emission spectroscopy metric (ESM) from Kempton et al (2018).
    
    Parameters
    ----------
    RpRs        : float or array_like
        Planet-to-star radius ratio
    Ts          : float or array_like
        Stellar temperature (K)
    aRs         : float or array_like
        Semi-major axis / stellar radius (unitless)
    kmag        : float or array_like
        Stellar K-band magnitude
    
    Returns
    -------
    esm         : float or array_like
        Emission spectroscopy metric
    '''
    Tp      = planetTeq(Ts, aRs, A=0, f=0.5)    # Dayside temperature
    Bp      = planck(7.5e-6, Tp)
    Bs      = planck(7.5e-6, Ts)
    FpFs    = planetStarEmission(RpRs, Bp, Bs)
    return 4.29 * FpFs * 10**(-kmag/5.)

class initObj:
    def __init__(self):
        pass

def loadRef(name, A=0, f=0.5):
    '''
    Load parameters for reference system.
    
    Parameters
    ----------
    name    : string
        planet name, currently limited to 'HD_209458_b' or 'WASP-43_b'
    A       : float
        Albedo (unitless), default is zero
    f       : float
        Heat redistribution factor (unitless), default is 0.5
    
    Returns
    -------
    ref     : object
        Instance of reference system class
    '''
    ref         = initObj()
    if name == 'HD_209458_b':
        # Parameters for HD 209458b
        ref.vmag    = 7.63                          # V-band magnitude
        ref.jmag    = 6.591                         # J-band magnitude
        ref.kmag    = 6.308                         # K-band magnitude
        ref.Rs      = 1.1780230                     # Stellar radius, Gaia DR2
        ref.Ts      = 6077.                         # Stellar temperature, Gaia DR2
        ref.RpRs    = 0.12247                       # Planet-star radius ratio, Stassun et al. 2017
        ref.aRs     = 8.814                         # Semi-major axis / stellar radius, Stassun et al. 2017
        ref.Mp      = 0.73                          # Planet mass, Stassun et al. 2017
        ref.dur     = 0.1277                        # Transit duration (days), computed using Seager et al. 2003
        ref.Rp      = ref.RpRs*ref.Rs*Rsun/Rjup     # Planet radius
        ref.gp      = planetGravity(ref.Mp, ref.Rp) # Planet gravity
        ref.Tp_eq   = planetTeq(ref.Ts, ref.aRs, A=0, f=1)          # Planet Equilibrium temperature
        ref.Tp_day  = planetTeq(ref.Ts, ref.aRs, A=0, f=0.5)        # Planet Dayside temperature
        ref.H       = planetScaleHeight(ref.Tp_eq, ref.gp, mu=2.3)  # Planet scale height
        ref.Bp15    = planck(1.5e-6, ref.Tp_day)    # Planet blackbody at 1.5 microns
        ref.Bp50    = planck(5.0e-6, ref.Tp_day)    # Planet blackbody at 5.0 microns
        #ref.Tp      = planetTeq(ref.Ts, ref.aRs, A, f)          # Planet temperature
        #ref.H       = planetScaleHeight(ref.Tp, ref.gp, mu=2.3) # Planet scale height
        #ref.Bp15    = planck(1.5e-6, ref.Tp)        # Planet blackbody at 1.5 microns
        #ref.Bp50    = planck(5.0e-6, ref.Tp)        # Planet blackbody at 5.0 microns
        ref.Bs15    = planck(1.5e-6, ref.Ts)        # Stellar blackbody at 1.5 microns
        ref.Bs50    = planck(5.0e-6, ref.Ts)        # Stellar blackbody at 5.0 microns
        ref.FpFs15  = planetStarEmission(ref.RpRs, ref.Bp15, ref.Bs15)
        ref.FpFs50  = planetStarEmission(ref.RpRs, ref.Bp50, ref.Bs50)
        ref.signal  = planetSignalSize(ref.H, ref.Rp, ref.Rs)
        ref.tsm     = TSM(ref.Rp, ref.Mp, ref.Rs, ref.Ts, ref.aRs, ref.jmag)
        ref.esm     = ESM(ref.RpRs, ref.Ts, ref.aRs, ref.kmag)
        print("Finished loading parameters for " + name)
    if name == 'WASP-43_b':
        # Parameters for WASP-43b
        ref.vmag    = 12.4                          # V-band magnitude
        ref.jmag    = 9.995                         # J-band magnitude
        ref.kmag    = 9.267                         # K-band magnitude
        ref.Rs      = 0.6629471                     # Stellar radius, Gaia DR2
        ref.Ts      = 4306.                         # Stellar temperature, Gaia DR2
        ref.RpRs    = 0.15942                       # Planet-star radius ratio, Hoyer et al. 2016
        ref.aRs     = 4.867                         # Semi-major axis / stellar radius, Hoyer et al. 2016
        ref.Mp      = 2.050                         # Planet mass, Bonomo et al. 2017
        ref.dur     = 0.0483                        # Transit duration (days), Hellier et al. 2011
        ref.Rp      = ref.RpRs*ref.Rs*Rsun/Rjup     # Planet radius
        ref.gp      = planetGravity(ref.Mp, ref.Rp) # Planet gravity
        ref.Tp_eq   = planetTeq(ref.Ts, ref.aRs, A=0, f=1)          # Planet Equilibrium temperature
        ref.Tp_day  = planetTeq(ref.Ts, ref.aRs, A=0, f=0.5)        # Planet Dayside temperature
        ref.H       = planetScaleHeight(ref.Tp_eq, ref.gp, mu=2.3)  # Planet scale height
        ref.Bp15    = planck(1.5e-6, ref.Tp_day)    # Planet blackbody at 1.5 microns
        ref.Bp50    = planck(5.0e-6, ref.Tp_day)    # Planet blackbody at 5.0 microns
        #ref.Tp      = planetTeq(ref.Ts, ref.aRs, A, f)          # Planet temperature
        #ref.H       = planetScaleHeight(ref.Tp, ref.gp, mu=2.3) # Planet scale height
        #ref.Bp15    = planck(1.5e-6, ref.Tp)        # Planet blackbody at 1.5 microns
        #ref.Bp50    = planck(5.0e-6, ref.Tp)        # Planet blackbody at 5.0 microns
        ref.Bs15    = planck(1.5e-6, ref.Ts)        # Stellar blackbody at 1.5 microns
        ref.Bs50    = planck(5.0e-6, ref.Ts)        # Stellar blackbody at 5.0 microns
        ref.FpFs15  = planetStarEmission(ref.RpRs, ref.Bp15, ref.Bs15)
        ref.FpFs50  = planetStarEmission(ref.RpRs, ref.Bp50, ref.Bs50)
        ref.signal  = planetSignalSize(ref.H, ref.Rp, ref.Rs)
        ref.tsm     = TSM(ref.Rp, ref.Mp, ref.Rs, ref.Ts, ref.aRs, ref.jmag)
        ref.esm     = ESM(ref.RpRs, ref.Ts, ref.aRs, ref.kmag)
        print("Finished loading parameters for " + name)
    return ref


