#Dependencies
import ciao_contrib.runtool as rt
from os import mkdir, path
from subprocess import Popen, run, DEVNULL
from glob import glob
from pycrates import read_file
from lightcurves import lc_sigma_clip
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pandas import read_csv
from astropy.units import keV, Kelvin,s, min
from astropy.constants import k_B
from time import time
#ciao must be installed (through conda preferred), caldb must be installed

###DEFINITIONS
chipnames=['i0', 'i1', 'i2', 'i3', 's0', 's1','s2', 's3', 's4', 's5'] #this list indexes chip ids to names
home=path.expanduser('~')
clusters=read_csv(home+'/chandra_obs/clusters.csv', index_col=0)


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
        run(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    return

def find_images(obj_name, obj_dir=_obj_dir_def):
    imgpath=obj_dir+obj_name_'/band1_thresh.img'
    exppath=obj_dir+obj_name_'/band1_thresh.expmap'
    bkgpath=obj_dir+obj_name_'/band1_thresh.bkg'
    if not path.isfile(imgpath):
        raise FileNotFoundError('Unable to find counts image')
    if not path.isfile(exppath):
        raise FileNotFoundError('Unable to find exposure map')
    if not path.isfile(bkgpath):
        raise FileNotFoundError('Unable to find background image')
    else:
        print('Image, Exposure Map, and Background files found.')
        return imgpath, exppath, bkgpath

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
    outpath=outdir+oufile
    cmd=['reproject_image', bkgpath, 'matchfile='+matchfile, 'outfile='+outpath, 'clobber=yes']
    shell(cmd)
    return outpath

def ds9(paths):
    #list of paths to fits files
    if type(paths)!=list:
        paths=[paths]
    cmd=['ds9']
    for p in paths:
        cmd.append(p)
        file, ext=path.splitext(p)
        cmd+=ds9_preferences[ext]
        cmd.append('-zoom to fit')
    cmd.append('-geometry 1920x1080')
    shell(cmd)
    return

###LEVEL 2 TOOLS
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

def make_weigths(obj_name, Emin=_Emin_def, Emax=_Emax_def, Tkev=None, nh22=None, obj_dir=_obj_dir_def):
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
    weightpath=obj_dir+obj_name+'_'+Emin+'_'+Emax+'.wts'
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

###LEVEL 3 TOOLS
def remove_flares(evtpath, chips=None, chip_paths=None, bin_time=_bin_time_def, plot=False):
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
    for n in range(len(chipnames)):
        try:
            cpath=chip_paths[str(n)]
            lcpath, gtipath=get_gti(cpath, bin_time, plot)
            cmd.append(chipnames[n]+'='+gtipath)
        except KeyError:
            cmd.append(chipnames[n]+'=none')   
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
    bracket='[@'+gtipath+']'
    rdtpath=dmcopy(rdtpath, rdtpath, bracket)
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

###LEVEL 5 TOOLS
def multi_obsid_image(obj_name, obsids=None, repro=False, update_weights=False, Emin=_Emin_def,
                      Emax=_Emax_def, bin_time=_bin_time_def, bin_pixel=_bin_pixel_def, bad_chip=8, 
                      obs_dir=_obs_dir_def, obj_dir=_obj_dir_def, display=True):
    #get exposure weights for object
    ti=clock()
    weightpath=obj_dir+obj_name+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+'.wts'
    if not path.isdir(weightpath) or update_weights:
        weightpath=make_weigths(obj_name,Emin, Emax)
    #Collect evtfile paths and bkg file paths in these dicts
    evt_paths={}
    bsky_paths={}
    rdt_paths={}
    if obsids==None:
        raise NotImplementedError('Obs ID Fetch Missing')
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

