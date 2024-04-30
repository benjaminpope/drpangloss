#! /usr/bin/env python

"""
@author: Anthony Soulain (University of Sydney), Rachel Cooper (STScI), Anand Sivaramakrishnan (STScI)
--------------------------------------------------------------------
implaneIA software
--------------------------------------------------------------------
OIFITS related function.
-------------------------------------------------------------------- 
"""

import datetime
import os
import copy
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt

import jax.numpy as jnp

plt.close('all')


list_color = ['#00a7b5', '#afd1de', '#055c63', '#ce0058', '#8a8d8f', '#f1b2dc']

def rad2mas(rad):
    return rad / u.milliarcsec.to(u.rad)


def GetWavelength(ins, filt):
    """ Get wavelengths information from using instrument and filter informations."""
    dic_filt = {'JWST': {'F277W': [2.776, 0.715],
                         'F380M': [3.828, 0.205],
                         'F430M': [4.286, 0.202],
                         'F480M': [4.817, 0.298]
                         }
                }

    wl = dic_filt[ins][filt][0]*1e-6
    e_wl = dic_filt[ins][filt][1]*1e-6

    return wl, e_wl


def Format_STAINDEX_V2(tab):
    """ Converts sta_index to save oifits in the appropriate format."""
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        if np.min(tab) == 0:
            line = np.array([ap1, ap2]) + 1 # RAC 2/2021
        else:
            line = np.array([ap1, ap2])
        sta_index.append(line)
    return sta_index


def Format_STAINDEX_T3(tab):
    """ Converts sta_index to save oifits in the appropriate format."""
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        ap3 = int(x[2])
        if np.min(tab) == 0:
            line = np.array([ap1, ap2, ap3]) + 1
        else:
            line = np.array([ap1, ap2, ap3])
        sta_index.append(line)
    return sta_index


def ApplyFlag(data, unit='arcsec'):
    """ Apply flag and convert to the appropriate units."""

    wl = data['OI_WAVELENGTH']['EFF_WAVE']
    uv_scale = {'m': 1,
                'rad': 1/wl,
                'arcsec': 1/wl/rad2mas(1e-3),
                'lambda': 1/wl/1e6}

    U = data['OI_VIS2']['UCOORD']*uv_scale[unit]
    V = data['OI_VIS2']['VCOORD']*uv_scale[unit]

    flag_v2 = np.invert(data['OI_VIS2']['FLAG'])
    V2 = data['OI_VIS2']['VIS2DATA'][flag_v2]
    e_V2 = data['OI_VIS2']['VIS2ERR'][flag_v2] * 1
    sp_freq_vis = data['OI_VIS2']['BL'][flag_v2] * uv_scale[unit]
    flag_cp = np.invert(data['OI_T3']['FLAG'])
    cp = data['OI_T3']['T3PHI'][flag_cp]
    e_cp = data['OI_T3']['T3PHIERR'][flag_cp]
    sp_freq_cp = data['OI_T3']['BL'][flag_cp] * uv_scale[unit]
    bmax = 1.2*np.max(np.sqrt(U**2+V**2))

    return U, V, bmax, V2, e_V2, cp, e_cp, sp_freq_vis, sp_freq_cp, wl, data['info']['FILT']


def save(dic, filename=None, datadir=None, verbose=False):
    """
    Save dictionary formatted data into a proper OIFITS (version 2) format file.
    Parameters:
    -----------
    `dic` {dict}:
        Dictionnary containing all extracted data (keys: 'OI_VIS2', 'OI_VIS', 'OI_T3', 'OI_WAVELENGTH', 'info'),\n
    `filename` {str}:
        By default None, the filename is constructed using informations included in the input dictionnary ('info'),\n
    """
    if dic is None:
        print('\nError save oifits : Wrong data format!')
        return None

    if datadir is None:
        datadir = 'Saveoifits/'
    if datadir[-1] != '/':
        datadir = datadir + '/'

    if not os.path.exists(datadir):
        print('### Create %s directory to save all requested Oifits ###' % datadir)
        os.system('mkdir %s' % datadir)
    
    if type(filename) != str:
        try:
            if len(dic['info']['MJD']) > 1:
                filename = '%s_%s_%s_%s_%s.oifits' % (dic['info']['TARGET'].replace(' ', ''),
                                                             dic['info']['INSTRUME'],
                                                             dic['info']['MASK'],
                                                             dic['info']['FILT'],
                                                             dic['info']['MJD'][0]) # if loaded from oifits it is a list
        except TypeError:
            filename = '%s_%s_%s_%s_%s.oifits' % (dic['info']['TARGET'].replace(' ', ''),
                                                     dic['info']['INSTRUME'],
                                                     dic['info']['MASK'],
                                                     dic['info']['FILT'],
                                                     dic['info']['MJD'])

    # ------------------------------
    #       Creation OIFITS
    # ------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :" % (filename))

    hdulist = fits.HDUList()
    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = datetime.datetime.now().strftime(
        format='%F')  # , 'Creation date'
    hdu.header['ORIGIN'] = 'STScI'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS']
    hdu.header['CONTENT'] = 'OIFITS2'
    hdu.header['TELESCOP'] = dic['info']['TELESCOP']
    hdu.header['INSTRUME'] = dic['info']['INSTRUME']
    hdu.header['OBSERVER'] = dic['info']['OBSERVER']
    hdu.header['OBJECT'] = dic['info']['OBJECT']
    hdu.header['INSMODE'] = dic['info']['INSMODE']
    hdu.header['FILT'] = dic['info']['FILT']
    hdu.header['ARRNAME'] = dic['info']['ARRNAME'] # Anand 9/2020
    hdu.header['MASK'] = dic['info']['ARRNAME'] # Anand 9/2020
    hdu.header['PA'] = dic['info']['PA'] # RC 1/2021
    # name of calibrator if applicable. RC 1/2021
    try:
        hdu.header['CALIB'] = dic['info']['CALIB']
    except KeyError:
        pass

    hdulist.append(hdu)
    # ------------------------------
    #        OI Wavelength
    # ------------------------------

    if verbose:
        print('-> Including OI Wavelength table...')
    data = dic['OI_WAVELENGTH']


    col1 = fits.Column(name='EFF_WAVE', format='1E',
                    unit='METERS', array=[data['EFF_WAVE']])
    col2 = fits.Column(name='EFF_BAND', format='1E',
                    unit='METERS', array=[data['EFF_BAND']])

    coldefs = fits.ColDefs([col1,col2])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    # Header
    hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
    hdu.header['OI_REVN'] = 2  # , 'Revision number of the table definition'
    # 'Name of detector, for cross-referencing'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdulist.append(hdu)  # Add current HDU to the final fits file.

    # ------------------------------
    #          OI Target
    # ------------------------------
    if verbose:
        print('-> Including OI Target table...')

    name_star = dic['info']['TARGET']

    customSimbad = Simbad()
    customSimbad.add_votable_fields('propermotions', 'sptype', 'parallax')

    # Add informations from Simbad:
    if name_star == 'UNKNOWN':
        ra, dec, spectyp = [0], [0], ['unknown']
        pmra, pmdec, plx = [0], [0], [0]
    else:
        try:
            query = customSimbad.query_object(name_star)
            coord = SkyCoord(query['RA'][0]+' '+query['DEC']
                             [0], unit=(u.hourangle, u.deg))
            ra, dec = [coord.ra.deg], [coord.dec.deg]
            spectyp, plx = query['SP_TYPE'], query['PLX_VALUE']
            pmra, pmdec = query['PMRA'], query['PMDEC']
        except TypeError:
            ra, dec, spectyp = [0], [0], ['unknown']
            pmra, pmdec, plx = [0], [0], [0]


    col1 = fits.Column(name='TARGET_ID', format='1I', array=[1])
    col2 = fits.Column(name='TARGET', format='16A', array=[name_star])
    col3 = fits.Column(name='RAEP0', format='1D', unit='DEGREES', array=ra)
    col4 = fits.Column(name='DECEP0', format='1D', unit='DEGREES', array=dec)
    col5 = fits.Column(name='EQUINOX', format='1E', unit='YEARS', array=[2000])
    col6 = fits.Column(name='RA_ERR', format='1D', unit='DEGREES', array=[0])
    col7 = fits.Column(name='DEC_ERR', format='1D', unit='DEGREES', array=[0])
    col8 = fits.Column(name='SYSVEL', format='1D', unit='M/S', array=[0])
    col9 = fits.Column(name='VELTYP', format='8A', array=['UNKNOWN'])
    col10 = fits.Column(name='VELDEF', format='8A', array=['OPTICAL'])
    col11 = fits.Column(name='PMRA', format='1D', unit='DEG/YR', array=pmra)
    col12 = fits.Column(name='PMDEC', format='1D', unit='DEG/YR', array=pmdec)
    col13 = fits.Column(name='PMRA_ERR', format='1D', unit='DEG/YR', array=[0])
    col14 = fits.Column(name='PMDEC_ERR', format='1D', unit='DEG/YR', array=[0])
    col15 = fits.Column(name='PARALLAX', format='1E', unit='DEGREES', array=plx)
    col16 = fits.Column(name='PARA_ERR', format='1E', unit='DEGREES', array=[0])
    col17 = fits.Column(name='SPECTYP', format='16A', array=spectyp)

    coldefs = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,
                            col10,col11,col12,col13,col14,col15,col16,col17])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_TARGET'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdulist.append(hdu)

    # ------------------------------
    #           OI Array
    # ------------------------------

    if verbose:
        print('-> Including OI Array table...')
    try:
        staxy = dic['info']['STAXY'] # these are the mask hole xy-coords as built (ctrs_inst)
    except KeyError:
        staxy = dic['OI_ARRAY']['STAXY']
    try:
        ctrs_eqt = dic['info']['CTRS_EQT']
    except KeyError:
        ctrs_eqt = dic['OI_ARRAY']['CTRS_EQT']

    N_ap = len(staxy)

    tel_name = ['A%i' % x for x in np.arange(N_ap)+1]
    sta_name = tel_name
    diameter = [0] * N_ap

    staxyz = []
    for x in staxy:
        a = list(x)
        line = [a[0], a[1], 0]
        staxyz.append(line)



    sta_index = np.arange(N_ap) + 1

    col1 = fits.Column(name='TEL_NAME', format='16A', array=tel_name)
    col2 = fits.Column(name='STA_NAME', format='16A', array=sta_name)
    col3 = fits.Column(name='STA_INDEX', format='1I', array=sta_index)
    col4 = fits.Column(name='DIAMETER', unit='METERS', format='1E', array=diameter)
    col5 = fits.Column(name='STAXYZ', unit='METERS', format='3D', array=staxyz)
    col8 = fits.Column(name='CTRS_EQT', unit='METERS', format='2D', array=ctrs_eqt) # for debugging

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_ARRAY'
    hdu.header['ARRAYX'] = float(0)
    hdu.header['ARRAYY'] = float(0)
    hdu.header['ARRAYZ'] = float(0)
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['FRAME'] = 'SKY'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'

    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS
    # ------------------------------

    if verbose:
        print('-> Including OI Vis table...')

    data = dic['OI_VIS']
    npts = len(dic['OI_VIS']['VISAMP'])

    sta_index = Format_STAINDEX_V2(sta_index)
    some_keys = ['TARGET_ID','TIME','MJD','INT_TIME']
    for akey in some_keys:
        try:
            if len(data[akey]) > 1:
                data[akey] = data[akey][0]
                if len(data[akey]) > 1:
                    data[akey] = data[akey][0]
        except TypeError:
            pass
    if len(data['VISAMP'].shape) == 1: # figure out if it's multi-slice or not
        nslice = 1
    else:
        nslice = data['VISAMP'].shape[1] # this would cause an error if not multi-slice
    col1 = fits.Column(name='TARGET_ID', format='1I',
                    array=[data['TARGET_ID']]*npts)
    col2 = fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=[data['TIME']]*npts)
    col3 = fits.Column(name='MJD', unit='DAY', format='1D',
                    array=[data['MJD']]*npts)
    col4 = fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts)
    col5 = fits.Column(name='VISAMP', format='%dD'%nslice, array=data['VISAMP'])
    col6 = fits.Column(name='VISAMPERR', format='%dD'%nslice, array=data['VISAMPERR'])
    col7 = fits.Column(name='VISPHI', format='%dD'%nslice, unit='DEGREES',
                    array=data['VISPHI'])
    col8 = fits.Column(name='VISPHIERR', format='%dD'%nslice, unit='DEGREES',
                    array=data['VISPHIERR'])
    col9 = fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD'])
    col10 = fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD'])
    col11 = fits.Column(name='STA_INDEX', format='2I', array=sta_index)
    col12 = fits.Column(name='FLAG', format='1L', array=data['FLAG'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9,
                            col10, col11, col12])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['EXTNAME'] = 'OI_VIS'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS2
    # ------------------------------
    if verbose:
        print('-> Including OI Vis2 table...')

    data = dic['OI_VIS2']
    npts = len(dic['OI_VIS2']['VIS2DATA'])

    some_keys = ['TARGET_ID', 'TIME', 'MJD', 'INT_TIME']
    for akey in some_keys:
        try:
            if len(data[akey]) > 1:
                data[akey] = data[akey][0]
                if len(data[akey]) > 1:
                    data[akey] = data[akey][0]
        except TypeError:
            pass
    col1 = fits.Column(name='TARGET_ID', format='1I',
                    array=[data['TARGET_ID']]*npts)
    col2 = fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=[data['TIME']]*npts)
    col3 = fits.Column(name='MJD', unit='DAY', format='1D',
                    array=[data['MJD']]*npts)
    col4 = fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts)
    col5 = fits.Column(name='VIS2DATA', format='%dD'%nslice, array=data['VIS2DATA'])
    col6 = fits.Column(name='VIS2ERR', format='%dD'%nslice, array=data['VIS2ERR'])
    col7 = fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD'])
    col8 = fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD'])
    col9 = fits.Column(name='STA_INDEX', format='2I', array=sta_index)
    col10 = fits.Column(name='FLAG', format='1L', array=data['FLAG'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9,
                            col10])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_VIS2'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI T3
    # ------------------------------
    if verbose:
        print('-> Including OI T3 table...')

    data = dic['OI_T3']
    npts = len(dic['OI_T3']['T3PHI'])

    sta_index = Format_STAINDEX_T3(sta_index)
    some_keys = ['TARGET_ID', 'TIME', 'MJD', 'INT_TIME']
    for akey in some_keys:
        try:
            if len(data[akey]) > 1:
                data[akey] = data[akey][0]
                if len(data[akey]) > 1:
                    data[akey] = data[akey][0]
        except TypeError:
            pass

    col1 = fits.Column(name='TARGET_ID', format='1I', array=[1]*npts)
    col2 = fits.Column(name='TIME', format='1D', unit='SECONDS', array=[0]*npts)
    col3 = fits.Column(name='MJD', format='1D', unit='DAY',
                    array=[data['MJD']]*npts)
    col4 = fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts)
    col5 = fits.Column(name='T3AMP', format='%dD'%nslice, array=data['T3AMP'])
    col6 = fits.Column(name='T3AMPERR', format='%dD'%nslice, array=data['T3AMPERR'])
    col7 = fits.Column(name='T3PHI', format='%dD'%nslice, unit='DEGREES',
                    array=data['T3PHI'])
    col8 = fits.Column(name='T3PHIERR', format='%dD'%nslice, unit='DEGREES',
                    array=data['T3PHIERR'])
    col9 = fits.Column(name='U1COORD', format='1D',
                    unit='METERS', array=data['U1COORD'])
    col10 = fits.Column(name='V1COORD', format='1D',
                    unit='METERS', array=data['V1COORD'])
    col11 = fits.Column(name='U2COORD', format='1D',
                    unit='METERS', array=data['U2COORD'])
    col12 = fits.Column(name='V2COORD', format='1D',
                    unit='METERS', array=data['V2COORD'])
    col13 = fits.Column(name='STA_INDEX', format='3I', array=sta_index)
    col14 = fits.Column(name='FLAG', format='1L', array=data['FLAG'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9,
                            col10, col11, col12, col13, col14])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header['EXTNAME'] = 'OI_T3'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK'] # Anand 9/2020
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #          Save file
    # ------------------------------
    #print(os.path.join(datadir,filename))
    hdulist.writeto(os.path.join(datadir,filename), overwrite=True)
    print('\n\n### OIFITS CREATED (%s).' % filename)
    del(hdu)
    del(hdulist)


def cp_indices(vis_sta_index, cp_sta_index):
    """Extracts indices for calculating closure phase from visibility and closure phase station indices"""
    i_cps1 = np.zeros(len(cp_sta_index),np.int32)
    i_cps2 = np.zeros(len(cp_sta_index),np.int32)
    i_cps3 = np.zeros(len(cp_sta_index),np.int32)

    for i in range(len(cp_sta_index)):
        i_cps1[i] = np.argwhere((cp_sta_index[i][0]==vis_sta_index[:,0])&(cp_sta_index[i][1]==vis_sta_index[:,1]))[0,0]
        i_cps2[i] = np.argwhere((cp_sta_index[i][1]==vis_sta_index[:,0])&(cp_sta_index[i][2]==vis_sta_index[:,1]))[0,0]
        i_cps3[i] = np.argwhere((cp_sta_index[i][0]==vis_sta_index[:,0])&(cp_sta_index[i][2]==vis_sta_index[:,1]))[0,0]
    return i_cps1,i_cps2,i_cps3 