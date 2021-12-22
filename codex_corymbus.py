#Dependencies
import ciao_contrib.runtool as rt
from os import mkdir, path
from subprocess import Popen, run, DEVNULL
from glob import glob
from lightcurves import lc_sigma_clip
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy.io import fits
from pandas import read_csv
from astropy.units import keV, Kelvin,s, min, arcmin, arcsec, count
from astropy.constants import k_B
from time import time
from math import floor
from pycrates import copy_colvals, read_file
import pyregion
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

import warnings
warnings.filterwarnings("ignore")
#ciao must be installed (through conda preferred), caldb must be installed

###DEFINITIONS
chipnames=['i0', 'i1', 'i2', 'i3', 's0', 's1','s2', 's3', 's4', 's5'] #this list indexes chip ids to names
ds9_preferences={'.img':['-scale log', '-cmap heat'], '.expmap':['-cmap a'], '.bkg':['-cmap b']}
home=path.expanduser('~')
clusters=read_csv('cluster_params.csv', index_col=0)


###DEFAULT PARAMETERS
_show_commands_def=False
_Emin_def=500
_Emax_def=3500
_bin_time_def=200
_bin_pixel_def=2
_obs_dir_def=home+'/chandra_obs/'
_obj_dir_def=home+'/objects/'


###BASIC FUNCTIONS
def clock():
    t=time()*s
    return t.to(min)

def bold(text):
    #returns bolded text to be printed
    return '\033[1m'+str(text)+'\033[0m'

def shell(cmd, display=False, capture=False, show_commands=_show_commands_def):
    #cmd is list of terminal parameters that will be separated by spaces
    if type(cmd)==list:
        cmd=' '.join(cmd)
    if show_commands:
        print('Running in shell:', cmd)
    if display and capture:
        raise ValueError('Display and capture cannot both be True') 
    if capture:
        out=run(cmd, shell=True, capture_output=True, text=True).stdout
        return out
    if display:
        run(cmd, shell=True)
    else:
        run(cmd, shell=True, stdout=DEVNULL)
    return

def open_fits(fitspath):
    hdu=fits.open(fitspath)
    header=hdu[0].header
    data=hdu[0].data
    hdu.close()
    return header, data

def disp_data(data,log=True):
    plt.figure(figsize=(16,16))
    if log:
        plt.imshow(data,norm=LogNorm())
    else:
        plt.imshow(data)
    plt.tight_layout()
    return

def find_images(obj_name, obsid=None, obj_dir=_obj_dir_def):
    paths={}
    message=['Found Files:']
    if obsid==None:
        obsid=''
        evtpath=obj_dir+obj_name+'/merged_evt.fits'
    else:
        obsid=str(obsid)+'_'
        evtpath=obj_dir+obj_name+'/'+obsid+'reproj_evt.fits'
    imgpath=obj_dir+obj_name+'/'+obsid+'band1_thresh.img'
    exppath=obj_dir+obj_name+'/'+obsid+'band1_thresh.expmap'
    bkgpath=obj_dir+obj_name+'/'+obsid+'band1_thresh.bkg'
    regpath=obj_dir+obj_name+'/'+obsid+'band1_thresh.reg'
    if path.isfile(imgpath):
        message.append('Image')
        paths['img']=imgpath
    if path.isfile(exppath):
        message.append('Exposure Map')
        paths['exp']=exppath
    if path.isfile(bkgpath):
        message.append('Background')
        paths['bkg']=bkgpath
    if path.isfile(evtpath):
        message.append('Event')
        paths['evt']=evtpath
    if path.isfile(regpath):
        message.append('Region')
        paths['reg']=regpath
    print(' '.join(message))
    return paths

def get_radial_profile(data,center=None,mask=None,num_bins=30,logscale=True, remove_negatives=False):
    """
    Creates an annulus map given a two-dimensional image.
    Finds the total of values in each annulus
    Mask is a boolean array of the same shape as the data.
    Scale can be 'lin' or 'log' if annulus spacing is linear or logarithmic.
    pixel size in arcseconds
    outputs sum in counts, 
    """
    class radial_profile:
        def __init__(self):
            self.sum=None
            self.area=None
            self.r=None
            self.r_map=None
    
    if remove_negatives:
        data[data<0]=0
    #prepare mask
    if np.all(mask==None):
        mask=np.ones_like(data,bool)
    mask*=np.isfinite(data)
    #define coordinates
    if center==None:
        npix,npiy=data.shape
        center=round(npix/2),round(npiy/2)
    yc,xc=center
    xdim,ydim=data.shape
    x1=np.arange(-xc,xdim-xc)
    y1=np.arange(-yc,ydim-yc)
    x,y=np.meshgrid(y1,x1)
    r_map=abs(x+1j*y)
    rmax=r_map[mask].max()
    #make bins
    if logscale:
        r_space=np.logspace(0,np.log10(rmax),num_bins)
    else:
        r_space=np.linspace(1,rmax,num_bins)
    nrad=len(r_space)
    #make data container
    profile=radial_profile()
    profile.sum=np.zeros(nrad)
    profile.area=np.zeros(nrad)
    profile.r=r_space
    profile.r_map=r_map
    #add 0 to r_space
    r_space=np.insert(r_space,0,0)
    #loop through bins
    for irad in range(nrad):
        minrad=r_space[irad]
        maxrad=r_space[irad+1]
        annulus=(r_map>=minrad)*(r_map<maxrad)*mask
        data_ann=data[annulus]
        profile.sum[irad]=data_ann.sum()
        profile.area[irad]=data_ann.size
    #apply units
    profile.sum*count
    profile.area*(arcsec**2).to(arcmin**2)
    profile.r*arcsec.to(arcmin)
    profile.r_map*arcsec.to(arcmin)
    return profile

def open_regions(regpath,imgpath):
    reg=pyregion.open(regpath)
    imghdu=fits.open(imgpath)
    regmask=reg.get_mask(hdu=imghdu[0])
    srcmask=(regmask==False)
    imghdu.close()
    return srcmask

def reshape(data,factor):
    #reduces resolution by a factor given
    rows=round(data.shape[0]/factor)
    cols=round(data.shape[1]/factor)
    return data.resize(rows,factor,cols,factor)

def beta_model(rad, S0, rc, beta):
    return S0*(1+(rad/rc)**2)**(-3*beta+0.5)


###LEVEL 1 TOOLS
def download(obsid, obs_dir=home+'/chandra_obs/'):
    #obsid can be a string or int
    cmd=['(cd', obs_dir+';', 'download_chandra_obsid', str(obsid)+')']
    shell(cmd, display=True)
    return

def punlearn(tool):
    cmd=['punlearn', tool]
    shell(cmd)
    return

def dmkeypar(evtpath, key):
    cmd=['dmkeypar', evtpath, key, 'echo+']
    param=shell(cmd, capture=True)
    return param.split('\n')[0]    

def dmcopy(evtpath, copypath, bracket=''):
    cmd=['dmcopy', '"'+evtpath+bracket+'"', copypath, 'clobber=yes']
    shell(cmd)
    return copypath

def get_chips(evtpath):
    #returns chip ids from reprocessed event file
    tab=read_file(evtpath)
    ids=tab.ccd_id.values
    ids=list(set(ids))
    return ids

def combine_images(img_paths, obj_name, matchfile, outfile, obj_dir=_obj_dir_def):
    outdir=obj_dir+obj_name+'/'
    bkgpath=','.join(img_paths)
    outpath=outdir+outfile
    cmd=['reproject_image', bkgpath, 'matchfile='+matchfile, 'outfile='+outpath, 'clobber=yes']
    shell(cmd)
    return outpath

def ds9(paths, regpath=None, ampersand=True):
    #list of paths to fits files
    if type(paths)!=list:
        paths=[paths]
    cmd=['ds9']
    for p in paths:
        cmd.append(p)
        file, ext=path.splitext(p)
        cmd+=ds9_preferences[ext]
        cmd.append('-zoom to fit')
    if regpath!=None:
        cmd.append('-regions load all '+regpath)
    cmd.append('-geometry 1920x1080')
    if ampersand==True:
        cmd.append('&')
    shell(cmd)
    return

def find_center(data, srcmask=None, factor=None, plot=True):
    if np.all(srcmask!=None):
        if np.all(srcmask==False):
            raise ValueError('Provided Mask covers entire field!')
        else:
            data=data*srcmask
    if factor==None:
        factor=floor(max(data.shape)/200)
        print('Resizing by',factor)
    rows=floor(data.shape[0]/factor)*factor
    cols=floor(data.shape[1]/factor)*factor
    data=data[:rows,:cols]
    data_resized=np.resize(data,(rows//factor, factor, cols//factor, factor))
    data_bin=data_resized.sum(axis=1).sum(axis=2)
    center_bin=np.where(data_bin==np.nanmax(data_bin))
    yc,xc=center_bin[0]*factor+round(factor/2), center_bin[1]*factor+round(factor/2)
    if plot==True:
        plt.figure(figsize=(16,16))
        plt.imshow(data, norm=LogNorm())
        plt.plot(xc,yc,'-rx')
    return xc,yc


###LEVEL 2 TOOLS
def get_mono_energy(evtpath):
    #returns mean energy in keV
    punlearn('dmstat')
    cmd=['dmstat', '"'+evtpath+'[cols energy]"', 'verbos=0']
    shell(cmd)
    cmd=['pget', 'dmstat', 'out_mean']
    out=shell(cmd, capture=True)
    mono=float(out.split('\n')[0])
    return mono/1000

def get_psf_map(imgpath,evtpath):
    #cant get sim_z for merged files?
    mono=get_mono_energy(evtpath)
    imgfile,ext=path.splitext(imgpath)
    psfpath=imgfile+'.psf'
    cmd=['mkpsfmap', imgpath, 'outfile='+psfpath, 'energy='+str(round(mono,2)),'ecf=0.393', 'clobber=yes']
    shell(cmd)
    return psfpath

def check_vf_pha(evtpath):
    dmode=dmkeypar(evtpath, 'DATAMODE')
    if dmode=='FAINT':
        return 'check_vf_pha=no'
    if dmode=='VFAINT':
        return 'check_vf_pha=yes'
    else:
        raise Exception('Something is wrong with the datamode!')
    return

def reprocess(obsid, obs_dir=_obs_dir_def):
    punlearn('chandra_repro')
    obsid=str(obsid)
    cmd=['chandra_repro', 'indir='+obs_dir+obsid+'/', 
         'outdir='+obs_dir+obsid+'/repro/',
         'clobber=yes', 'set_ardlib=no']
    shell(cmd)
    return

def start_bpix(obsid, obs_dir=_obs_dir_def):
    #set bad pixel file for reprocessed observation
    obsid=str(obsid)
    path=glob(obs_dir+obsid+'/repro/*repro_bpix1.fits')
    if len(path)==1:
        punlearn('ardlib')
        cmd=['acis_set_ardlib', path[0], 'absolutepath=yes']
        shell(cmd)
    else:
        raise Exception('Too many files found')
    return

def get_event_file(obsid, obs_dir=_obs_dir_def):
    #find evt file for reprocessed observation
    obsid=str(obsid)
    try:
        path=glob(obs_dir+obsid+'/repro/*repro_evt2.fits')
        evtpath=obs_dir+obsid+'/'+obsid+'.evt'
        evtpath=dmcopy(path[0],evtpath)
        print('evt2 file found')
    except:
        try:
            path=glob(obs_dir+obsid+'/repro/*repro_evt1.fits')
            evtpath=obs_dir+obsid+'/'+obsid+'.evt'
            evtpath=dmcopy(path[0],evtpath)
            print('evt1 file found')
        except:
            print('no reprocessed event file found')
            evtpath=None
    return evtpath

def energy_filter(evtpath, Emin=_Emin_def, Emax=_Emax_def):
    #Filter event file with energy in eV
    evtfile,ext=path.splitext(evtpath)
    filterpath=evtfile+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+ext
    bracket='[energy='+str(Emin)+':'+str(Emax)+']'
    return dmcopy(evtpath, filterpath, bracket=bracket)

def split_chips(evtpath, chips=None, Emin=_Emin_def, Emax=_Emax_def):
    #returns dictionary. Keys are chip ids as strings,
    #values are paths to chip evt2 files
    evtfile,ext=path.splitext(evtpath)
    if chips==None:
        chips=get_chips(evtpath)
    chip_paths={}
    for c in chips:
        c=str(c)
        cpath=evtfile+'_c'+c+ext
        bracket='[ccd_id='+c+']'
        dmcopy(evtpath, cpath, bracket)
        chip_paths[c]=cpath
    return chip_paths

def get_lc(evtpath, bin_time=_bin_time_def,plot=False):
    evtfile, ext=path.splitext(evtpath)
    lcpath=evtfile+'.lc'
    bracket='[bin time=::'+str(bin_time)+']'
    cmd=['dmextract', '"'+evtpath+bracket+'"', 'outfile='+lcpath, 'opt=ltc1', 'clobber=yes']
    shell(cmd,display=True)
    return lcpath

def get_gti(evtpath, bin_time=_bin_time_def,plot=False):
    #evtpath is a path to an event file ending in .fits
    #returns path to lightcurve and gti file
    evtfile, ext=path.splitext(evtpath)
    lcpath=evtfile+'.lc'
    gtipath=evtfile+'.gti'
    bracket='[bin time=::'+str(bin_time)+']'
    cmd=['dmextract', '"'+evtpath+bracket+'"', 'outfile='+lcpath, 'opt=ltc1', 'clobber=yes']
    shell(cmd,display=True)
    lc_sigma_clip(lcpath, gtipath, plot=plot,verbose=0)
    return lcpath, gtipath

def remove_chip(evtpath, ccd_id=8):
    ccd_id=str(ccd_id)
    evtfile,ext=path.splitext(evtpath)
    filterpath=evtfile+'_no'+ccd_id+ext
    bracket='[exclude ccd_id='+ccd_id+']'
    return dmcopy(evtpath, filterpath, bracket)

def exposure_correct(bkgpath, imgpath, outpath=None):
    #make bkg exposure equal to img
    bkgfile, ext=path.splitext(bkgpath)
    bkgfits=fits.open(bkgpath)
    bkg_arr=bkgfits[0].data
    bkg_head=bkgfits[0].header
    bkgfits.close()
    bkg_exp=float(dmkeypar(bkgpath,'EXPOSURE'))
    img_exp=float(dmkeypar(imgpath,'EXPOSURE'))
    scaled_arr=bkg_arr*img_exp/bkg_exp
    bkg_head['EXPOSURE']=img_exp
    hdu=fits.PrimaryHDU(scaled_arr, header=bkg_head)
    if outpath==None:
        outpath=bkgfile+'_scaled'+ext
    hdu.writeto(outpath, overwrite=True)
    return outpath

def make_weights(obj_name, Emin=_Emin_def, Emax=_Emax_def, Tkev=None, nh22=None, obj_dir=_obj_dir_def):
    #assumes bremsstrahlung, needs nh22 and T in keV
    #if no Temperature or absorption is provided, lookup in parameter table
    Emin=str(Emin/1e3)
    Emax=str(Emax/1e3)
    if Tkev==None:
        Tkev=clusters['Tkev'][obj_name]
    if nh22==None:
        nh22=float(clusters['nh'][obj_name])/1e22
    TK=str((Tkev*keV/k_B).decompose().value)
    punlearn('make_instmap_weights')
    weightpath=obj_dir+obj_name+'/'+obj_name+'_'+Emin+'_'+Emax+'.wts'
    paramvals="paramvals='gal.nh="+str(nh22)+';br.temperature='+TK+"'"
    cmd=['make_instmap_weights', weightpath,"'xsphabs.gal*bremsstrahlung.br'", 
         paramvals, 'emin='+Emin, 'emax='+Emax, 'ewidth=0.1', 'clobber=yes']
    shell(cmd)
    return weightpath

def merge_events(evt_paths, obj_name, bin_pixel=_bin_pixel_def, Emin=_Emin_def, Emax=_Emax_def,
                 obj_dir=_obj_dir_def):
    #take list of clean evt files, produce exposure map and counts image
    #make weights file if none exists
    weightpath=obj_dir+obj_name+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+'.wts'
    objroot=obj_dir+obj_name+'/'
    punlearn('merge_obs')
    
    cmd=['merge_obs', ','.join(evt_paths),"outroot="+objroot, 'binsize='+str(bin_pixel),
         "bands='"+weightpath+"'",'clobber=yes', "expmapthresh='2%'"]
    shell(cmd)
    imgpath=objroot+'band1_thresh.img'
    exppath=objroot+'band1_thresh.expmap'
    fluxpath=objroot+'band1_flux.img'
    return imgpath, exppath, fluxpath

def remove_points(imgpath,exppath=None, psfpath=None, method='wavdetect'):
    punlearn(method)
    imgfile,ext=path.splitext(imgpath)
    srcpath=imgfile+'.src'
    scellpath=imgfile+'.scell'
    reconpath=imgfile+'_recon.img'
    defnbkgpath=imgfile+'.nbgd'
    regpath=imgfile+'.reg'
    cmd=[method, 'infile='+imgpath]
    if exppath!=None:
        cmd.append('expfile='+exppath)
    if psfpath!=None:
        cmd.append('psffile='+psfpath) 
    cmd+=['outfile='+srcpath, 'scellfile='+scellpath,
          'imagefile='+reconpath, 'defnbkgfile='+defnbkgpath,
          'regfile='+regpath, 'clobber=yes']
    shell(cmd)
    ds9(imgpath, regpath, ampersand=False)
    return srcpath, scellpath, reconpath, defnbkgpath, regpath

def surface_brightness_profile(imgdata,expdata,bkgdata=None,srcmask=None,plot=False):
    center=find_center(imgdata,srcmask,plot=False)
    #calculate radial profiles
    bscprof=get_radial_profile(imgdata-bkgdata,center,srcmask,logscale=True)
    expprof=get_radial_profile(expdata,center,srcmask,logscale=True)
    bkgprof=get_radial_profile(bkgdata,center,srcmask,logscale=True)
    
    sb_prof=uarray(bscprof.sum, np.sqrt(bscprof.sum))/expprof.sum
    pn_prof=uarray(bkgprof.sum, np.sqrt(bkgprof.sum))/expprof.sum
    rad=bscprof.r
    r_map=bscprof.r_map
    if plot:
        plt.figure(figsize=(12,6))
        plt.errorbar(rad, nominal_values(sb_prof), yerr=std_devs(sb_prof), label='Surface Brightness', ls='none', marker='o')
        plt.errorbar(rad, nominal_values(pn_prof), yerr=std_devs(pn_prof), label='Poisson Noise', ls='none', marker='o')
        plt.xlabel('Radius [arcmin]')
        plt.ylabel(r'$L_X$ [$counts/s/cm^2$]')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.title(obj)
    return rad, sb_prof, pn_prof, r_map


###LEVEL 3 TOOLS
def remove_flares(evtpath, chips=None, chip_paths=None, bin_time=_bin_time_def, plot=True):
    #takes reprocessed evt file path
    #returns path to filtered gti using all chips
    #user can provide list of chips and dict of chip_paths
    #get chip ids if none are provided
    evtfile,ext=path.splitext(evtpath)
    if chips==None:
        chips=get_chips(evtpath)
    #get paths to chip evt2 files if none are provided
    if chip_paths==None:
        chip_paths=split_chips(evtpath, chips)
    cmd=['multi_chip_gti', 'infile='+evtpath]
    #loop through chip ids, get light curve and gti files for each
    if plot:
        num_chips=len(chips)
        fig=plt.figure(figsize=(16,4*num_chips))
        ichip=1
    for n in range(len(chipnames)):
        try:
            cpath=chip_paths[str(n)]
            lcpath, gtipath=get_gti(cpath, bin_time)
            cmd.append(chipnames[n]+'='+gtipath)
            if plot:
                ax=fig.add_subplot(num_chips,1,ichip)
                ichip+=1
                
                lc=read_file(lcpath)
                cr=copy_colvals(lc,'COUNT_RATE')
                time=copy_colvals(lc,'TIME')
                time=time[cr!=0]
                cr=cr[cr!=0]
                ax.scatter(time,cr,color='red')
                
                cfile,ext=path.splitext(cpath)
                deflaredpath=cfile+'_clean'+ext
                bracket='[@'+gtipath+']'
                dmcopy(cpath, deflaredpath, bracket)
                
                lcpath=get_lc(deflaredpath, bin_time)
                lc=read_file(lcpath)
                cr=copy_colvals(lc,'COUNT_RATE')
                time=copy_colvals(lc,'TIME')
                time=time[cr!=0]
                cr=cr[cr!=0]
                std=np.std(cr)
                mean=np.mean(cr)
                ax.hlines([mean+std*3,mean-std*3], time.min(), time.max(),linestyle='dashed')
                ax.scatter(time,cr,color='lime')
                ax.set_xlim(time.min(),time.max())
                plt.ylabel('Count Rate [cts/s]')
                plt.savefig(evtfile+'_lc.png', facecolor='white')
        except KeyError:
            cmd.append(chipnames[n]+'=none')   
    plt.xlabel('Time [s]')
    multigti=evtfile+'.gti'
    cmd.append('out='+multigti)
    cmd.append('clobber=yes')
    #run multi_chip_gti using chip gtis
    punlearn('multi_chip_gti')
    shell(cmd)
    deflaredpath=evtfile+'_clean'+ext
    bracket='[@'+multigti+']'
    deflaredpath=dmcopy(evtpath, deflaredpath, bracket)
    print('Deflaring Removed', 
          bold(float(dmkeypar(evtpath,'EXPOSURE'))-float(dmkeypar(deflaredpath,'EXPOSURE'))),'s')
    return deflaredpath, multigti

def remove_flares_conservative(evtpath, chips=None, chip_paths=None, bin_time=_bin_time_def, plot=True):
    #takes reprocessed evt file path
    #returns path to filtered gti using all chips
    #user can provide list of chips and dict of chip_paths
    #get chip ids if none are provided
    evtfile,ext=path.splitext(evtpath)
    lcpath,gtipath=get_gti(evtpath,bin_time)
    deflaredpath=evtfile+'_clean'+ext
    if chips==None:
        chips=get_chips(evtpath)
    #get paths to chip evt2 files if none are provided
    if chip_paths==None:
        chip_paths=split_chips(evtpath, chips)
    for n in range(len(chipnames)):
        try:
            cpath=chip_paths[str(n)]
            clcpath, cgtipath=get_gti(cpath, bin_time)
            bracket='[@'+cgtipath+']'
            evtpath=dmcopy(evtpath, deflaredpath, bracket)
        except KeyError:
            pass
    if plot:
        plt.figure(figsize=(16,6))
        lc=read_file(lcpath)
        cr=copy_colvals(lc,'COUNT_RATE')
        time=copy_colvals(lc,'TIME')
        time=time[cr!=0]
        cr=cr[cr!=0]
        plt.scatter(time,cr,color='red')
        
        lcpath=get_lc(deflaredpath, bin_time)
        lc=read_file(lcpath)
        cr=copy_colvals(lc,'COUNT_RATE')
        time=copy_colvals(lc,'TIME')
        time=time[cr!=0]
        cr=cr[cr!=0]
        std=np.std(cr)
        mean=np.mean(cr)
        plt.hlines([mean+std*3,mean-std*3], time.min(), time.max(),linestyle='dashed')
        plt.scatter(time,cr,color='lime')
        plt.xlim(time.min(),time.max())
        plt.xlabel('Time [s]')
        plt.ylabel('Count Rate [cts/s]')
        plt.savefig(evtfile+'_lc.png', facecolor='white')
    return evtpath

def get_backgrounds(evtpath, gtipath, Emin=_Emin_def, Emax=_Emax_def):
    #evt file path as str ending in .fits
    #first get blanksky .bkg file
    evtfile,ext=path.splitext(evtpath)
    punlearn('blanksky')
    bskypath=evtfile+'.bsky'
    blanksky=['blanksky', 'evtfile='+evtpath, 'outfile='+bskypath, 'clobber=yes']
    shell(blanksky)
    #then get readout .rdt file
    rdtpath=evtfile+'.rdt'
    dirpath=path.dirname(evtpath)
    readout=['readout_bkg', dirpath, rdtpath, check_vf_pha(evtpath), 'clobber=yes']
    shell(readout)
    #filter bsky and rdt file
    bskypath=energy_filter(bskypath, Emin, Emax)
    rdtpath=energy_filter(rdtpath,Emin,Emax)
    #bracket='[@'+gtipath+']'
    #rdtpath=dmcopy(rdtpath, rdtpath, bracket)
    return bskypath, rdtpath

def scale_background(bskypath, rdtpath, imgpath, obj_name, obj_dir=_obj_dir_def):
    imgfile,ext=path.splitext(imgpath)
    #use blanksky_img to correct particle background
    cmd=['blanksky_image', 'bkgfile='+bskypath, 'outroot='+imgfile,
         'imgfile='+imgpath, 'mode=h', 'clobber=yes']
    shell(cmd)
    bskyimg=imgfile+'_particle_bgnd.img'
    #manually edit exposure of readout image
    rdtpath=combine_images([rdtpath], obj_name, imgpath, 'band1_thresh.rdt', obj_dir)
    rdtimg=exposure_correct(rdtpath, imgpath, imgfile+'_readout.img')
    #add scaled background files together
    rdtfits=fits.open(rdtimg)[0]
    bskyfits=fits.open(bskyimg)[0]
    rdt_header=rdtfits.header
    rdt_arr=rdtfits.data
    bsky_arr=bskyfits.data
    bkg_arr=rdt_arr+bsky_arr
    hdu=fits.PrimaryHDU(bkg_arr,header=rdt_header)
    bkgpath=imgfile+'.bkg'
    hdu.writeto(bkgpath, overwrite=True)
    return bkgpath

def fit_to_beta_model(rad, sb, pn=None, plot=True, plotpath=None):
    mean=nominal_values(sb)
    sigma=std_devs(sb)
    #if PN is provided, remove all points after pn dominates
    if np.any(pn!=None):
        for i in range(len(pn)):
            if nominal_values(pn)[i]>mean[i]:
                cut=i
                break
            else:
                pass
    popt, pcov=curve_fit(beta_model, rad[:cut], mean[:cut], [mean[0],1,1], sigma[:cut])
    S0, rc, beta=popt
    S0_sig, rc_sig, beta_sig=np.sqrt(np.diag(pcov))
    S0_fit=ufloat(S0,S0_sig)
    rc_fit=ufloat(rc,rc_sig)
    beta_fit=ufloat(beta, beta_sig)
    if plot:
        plt.figure(figsize=(12,6))
        plt.errorbar(rad,mean,yerr=sigma,label='Surface Brightness Profile',ls='none',marker='o')
        if np.any(pn!=None):
            plt.errorbar(rad, nominal_values(pn), yerr=std_devs(pn), label='Poisson Noise',ls='none',marker='o')
        plt.plot(rad,beta_model(rad,S0,rc,beta),label='Fitted Beta Model')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'Surface Brightness [$cts/s/cm^2$]')
        plt.xlabel(r'Radius [$arcmin$]')
        plt.legend()
        plt.annotate(' S0='+str(S0_fit)+'}\n rc='+str(rc_fit)+'\n beta='+str(beta_fit),
                    xy=(0.4, 0.05), xycoords='axes fraction', fontsize=16)
        if plotpath!=None:
            plt.savefig(plotpath)
    return S0_fit, rc_fit, beta_fit


###LEVEL 4 TOOLS
def process_obsid(obsid, repro=False, update_weights=False, obs_dir=_obs_dir_def, obj_dir=_obj_dir_def,
                  Emin=_Emin_def, Emax=_Emax_def, bin_time=_bin_time_def, 
                  bin_pixel=_bin_pixel_def, bad_chip=8):
    print('Processing ObsID', obsid)
    ti=clock()
    obsid=str(obsid)
    #download and reprocess obsid if requested
    if not path.isdir(obs_dir+obsid):
        print('Obsid '+obsid+' missing, downloading from archive.')
        download(obsid, obs_dir)
    if not path.isdir(obs_dir+obsid+'/repro') or repro:
        print('Reprocessing Obsid '+obsid)
        reprocess(obsid, obs_dir)
    start_bpix(obsid, obs_dir=obs_dir)
    #format the reprocessed event file how we want it
    evtpath=get_event_file(obsid, obs_dir)
    print('Event File:',bold(evtpath))
    chip_ids=get_chips(evtpath)
    if bad_chip in chip_ids:
        evtpath=remove_chip(evtpath, bad_chip)
        chip_ids.remove(bad_chip)
        print('Bad Chip Removed:', bold(evtpath))
    #split into chips, deflare
    chip_paths=split_chips(evtpath, chip_ids)
    evtpath, gtipath=remove_flares(evtpath, chip_ids, chip_paths, bin_time)
    print('Deflared:',bold(evtpath))
    #make background file
    bskypath, rdtpath=get_backgrounds(evtpath, gtipath, Emin, Emax)
    print('Backgrounds:', bold(bskypath), bold(rdtpath))
    #apply energy filter to event file
    evtpath=energy_filter(evtpath, Emin, Emax)
    print('Energy Filter:',bold(evtpath))
    tf=clock()
    print('Processing Obsid', obsid, 'took', tf-ti)
    print('\n')
    return evtpath, bskypath, rdtpath

def remove_beta_model(obj,plot=True,redo_src=False):
    paths=find_images(obj)
    imgfile,ext=path.splitext(paths['img'])
    #unpack fits files
    imghead,imgdata=open_fits(paths['img'])
    exphead,expdata=open_fits(paths['exp'])
    bkghead,bkgdata=open_fits(paths['bkg'])
    #locate point source mask
    if not redo_src:
        try:
            reg=paths['reg']
        except KeyError:
            src,scell,recon,defnbkg,reg=remove_points(paths['img'], paths['exp'])
    else:
        src,scell,recon,defnbkg,reg=remove_points(paths['img'], paths['exp'])
    srcmask=open_regions(reg,paths['img'])
    #calculate surface brightness profile, fit to beta model
    rad, sb, pn, r_map=surface_brightness_profile(imgdata,expdata,bkgdata,srcmask)
    params=fit_to_beta_model(rad, sb, pn, plotpath=imgfile+'_betamod.png')
    S0,rc,beta=nominal_values(params)
    betamap=beta_model(r_map, S0, rc, beta)
    fluxmap=(imgdata-bkgdata)/expdata
    fluxmap[fluxmap<0]=0
    if plot:
        disp_data(fluxmap/betamap*srcmask)
    return fluxmap/betamap


###LEVEL 5 TOOLS
def multi_obsid_image(obj_name, obsids=None, repro=False, update_weights=False, Emin=_Emin_def,
                      Emax=_Emax_def, bin_time=_bin_time_def, bin_pixel=_bin_pixel_def, bad_chip=8, 
                      obs_dir=_obs_dir_def, obj_dir=_obj_dir_def, display=True):
    #get exposure weights for object
    ti=clock()
    weightpath=obj_dir+obj_name+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+'.wts'
    if not path.isdir(weightpath) or update_weights:
        weightpath=make_weights(obj_name,Emin, Emax)
    #Collect evtfile paths and bkg file paths in these dicts
    evt_paths={}
    bsky_paths={}
    rdt_paths={}
    if obsids==None:
        try:
            obsids=clusters['obsids'][obj_name].split(',')
        except:
            raise NotImplementedError('No Obsids Found')
    for o in obsids:
        o=str(o)
        processed=process_obsid(o, repro, update_weights, obs_dir, obj_dir,
                                Emin, Emax, bin_time, bin_pixel, bad_chip)
        evt_paths[o], bsky_paths[o], rdt_paths[o]=processed
    evt_paths=list(evt_paths.values())
    img, exp, flux=merge_events(evt_paths, obj_name, bin_pixel, Emin, Emax, obj_dir)
    print('Counts Image:', bold(img))
    print('Exposure Map:', bold(exp))
    print('Flux Image:', bold(flux))
    bkg_paths=[]
    for o in obsids:
        o=str(o)
        imgpath=obj_dir+obj_name+'/'+o+'_band1_thresh.img'
        bkgpath=scale_background(bsky_paths[o], rdt_paths[o], imgpath, obj_name)
        bkg_paths.append(bkgpath)
    bkg=combine_images(bkg_paths, obj_name, img, 'band1_thresh.bkg', obj_dir)
    print('Merged Backgrounds:', bold(bkg))
    tf=clock()
    print('Imaging', obj_name, 'took', tf-ti)
    if display==True:
        ds9([img, exp, flux, bkg])
    return img, exp, flux, bkg

