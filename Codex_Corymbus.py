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
from astropy.units import keV, Kelvin
from astropy.constants import k_B

#definitions
home=path.expanduser('~')
chipnames=['i0', 'i1', 'i2', 'i3', 's0', 's1','s2', 's3', 's4', 's5'] #this list indexes chip ids to names
clusters=read_csv('/home/andy/chandra_obs/clusters.csv', index_col=0)
ds9_preferences={'.img':['-scale log', '-cmap heat'], '.expmap':['-cmap a'], '.bkg':['-cmap b']}
#Basic functions
def bold(text):
    #prints bolded text
    print('\033[1m')
    print(text)
    print('\033[0m')
    return


def shell(cmd, display=False, capture=False):
    #cmd is list of terminal parameters that will be separated by spaces
    if type(cmd)==list:
        cmd=' '.join(cmd)
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


def punlearn(tool):
    cmd=['punlearn', tool]
    shell(cmd)
    return


def dmkeypar(evtpath, key):
    cmd=['dmkeypar', evtpath, key, 'echo+']
    param=shell(cmd, capture=True)
    return param.split('\n')[0]


def dmcopy(evtpath, copypath, bracket=''):
    punlearn('dmcopy')
    cmd=['dmcopy', '"'+evtpath+bracket+'"', copypath, 'clobber=yes']
    shell(cmd)
    return copypath


def check_vf_pha(evtpath):
    dmode=dmkeypar(evtpath, 'DATAMODE')
    if dmode=='FAINT':
        return 'check_vf_pha=no'
    if dmode=='VFAINT':
        return 'check_vf_pha=yes'
    else:
        raise Exception('Something is wrong with the datamode!')
    return


def start_bpix(obsid, obs_dir=home+'/chandra_obs/'):
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


#Tier 3 functions
def download(obsid, obs_dir=home+'/chandra_obs/'):
    #obsid can be a string or int
    cmd=['(cd', obs_dir+';', 'download_chandra_obsid', str(obsid)+')']
    shell(cmd, display=True)
    return


def reprocess(obsid, obs_dir=home+'/chandra_obs/'):
    punlearn('chandra_repro')
    obsid=str(obsid)
    cmd=['chandra_repro', 'indir='+obs_dir+obsid+'/', 
         'outdir='+obs_dir+obsid+'/repro/',
         'clobber=yes', 'set_ardlib=no']
    shell(cmd)
    return


def get_event_file(obsid, obs_dir=home+'/chandra_obs/'):
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


def remove_chip(evtpath, ccd_id=8):
    punlearn('dmcopy')
    ccd_id=str(ccd_id)
    evtfile,ext=path.splitext(evtpath)
    filterpath=evtfile+'_no'+ccd_id+ext
    bracket='[exclude ccd_id='+ccd_id+']'
    return dmcopy(evtpath, filterpath, bracket)


def get_chips(evtpath):
    #returns chip ids from reprocessed event file
    tab=read_file(evtpath)
    ids=tab.ccd_id.values
    ids=list(set(ids))
    return ids


def split_chips(evtpath, chips=None, Emin=300, Emax=12000):
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


def get_gti(evtpath, bin_time=200):
    #evtpath is a path to an event file ending in .fits
    #returns path to lightcurve and gti file
    evtfile, ext=path.splitext(evtpath)
    lcpath=evtfile+'.lc'
    gtipath=evtfile+'.gti'
    bracket='[bin time=::'+str(bin_time)+']'
    cmd=['dmextract', '"'+evtpath+bracket+'"', 'outfile='+lcpath, 'opt=ltc1', 'clobber=yes']
    shell(cmd,display=True)
    lc_sigma_clip(lcpath, gtipath, plot=False, verbose=0)
    return lcpath, gtipath


def image_events(evtpath, bin_pixel=8):
    evtfile,ext=path.splitext(evtpath)
    binnedpath=evtfile+'_img'+str(bin_pixel)+ext
    bracket='[bin sky='+str(bin_pixel)+']'
    return dmcopy(evtpath, binnedpath, bracket)


def exposure_correct(imgpath, evtpath):
    #take input image, divide by its exposure, multiply by exposure of reference image
    input_file, ext=path.splitext(imgpath)
    input_fits=fits.open(imgpath)
    ref_fits=fits.open(evtpath)
    input_header=input_fits[0].header
    input_exp=input_header['EXPOSURE']
    input_arr=input_fits[0].data
    ref_exp=dmkeypar(evtpath, 'EXPOSURE')
    input_fits.close()
    ref_fits.close()
    corrected_arr=input_arr*float(ref_exp)/float(input_exp)
    hdu=fits.PrimaryHDU(corrected_arr, header=input_header)
    outpath=input_file+'scaled'+ext
    hdu.writeto(outpath, overwrite=True)
    return outpath


def energy_filter(evtpath, Emin=300, Emax=12000):
    #Filter event file with energy in eV
    evtfile,ext=path.splitext(evtpath)
    filterpath=evtfile+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+ext
    bracket='[energy='+str(Emin)+':'+str(Emax)+']'
    return dmcopy(evtpath, filterpath, bracket=bracket)


#Tier 2 functions
def remove_flares(evtpath, chips=None, chip_paths=None, bin_time=200):
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
            lcpath, gtipath=get_gti(cpath, bin_time)
            cmd.append(chipnames[n]+'='+cpath)
        except KeyError:
            cmd.append(chipnames[n]+'=none')   
    gtipath=evtfile+'.gti'
    cmd.append('out='+gtipath)
    cmd.append('clobber=yes')
    #run multi_chip_gti using chip gtis
    #punlearn('multi_chip_gti')
    shell(cmd)
    deflaredpath=evtfile+'_clean'+ext
    bracket='[@'+gtipath+']'
    return dmcopy(evtpath, deflaredpath, bracket), gtipath


def get_backgrounds(evtpath, gtipath, Emin=300, Emax=12000, bin_pixel=8):
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
    efilterbsky=energy_filter(bskypath, Emin, Emax)
    efilterrdt=energy_filter(rdtpath,Emin,Emax)
    bracket='[@'+gtipath+']'
    cleanrdt=dmcopy(efilterrdt, evtfile+'.rdt', bracket)
    #image clean rdt and energy filtered bsky
    imgbsky=image_events(efilterbsky, bin_pixel)
    imgrdt=image_events(cleanrdt, bin_pixel)
    return imgbsky, imgrdt


def apply_backgrounds(backgrounds, evtpath):
    #backgrounds should be iterable of paths to files
    #corrects exposure and combines backgrounds
    #output fits file is background image with header of first element in backgrounds
    corr_paths=[]
    corr_arrs=[]
    corr_heads=[]
    for bkg in backgrounds:
        corrpath=exposure_correct(bkg, evtpath)
        corr_paths.append(corrpath)
        corrfits=fits.open(corrpath)
        corrarr=corrfits[0].data
        corrhead=corrfits[0].header
        corrfits.close()
        corr_arrs.append(corrarr)
        corr_heads.append(corrhead)
    bkgarr=sum(corr_arrs)
    hdu=fits.PrimaryHDU(bkgarr,header=corr_heads[0])
    imagefile, ext=path.splitext(evtpath)
    bkgpath=imagefile+'.bkg'
    hdu.writeto(bkgpath,overwrite=True)
    return bkgpath


#Tier 1 functions
def make_weigths(obj_name, Emin=300, Emax=12000, Tkev=None, nh22=None, obj_dir=home+'/objects/'):
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


def process_obsid(obsid, obj_name, repro=False, update_weights=False, obs_dir=home+'/chandra_obs/', 
                obj_dir=home+'/objects/', Emin=300, Emax=12000, bin_time=200, bin_pixel=8, bad_chip=8):
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
    bold(evtpath)
    #filter event file 
    if bad_chip!=None:
        evtpath=remove_chip(evtpath, bad_chip)
        bold(evtpath)
    #split into chips, deflare
    chip_ids=get_chips(evtpath)
    chip_paths=split_chips(evtpath, chip_ids)
    evtpath, gtipath=remove_flares(evtpath, chip_ids, chip_paths, bin_time)
    bold(evtpath)
    #make background file
    backgrounds=get_backgrounds(evtpath, gtipath, Emin, Emax, bin_pixel)
    bold(backgrounds)
    #apply energy filter to event file
    evtpath=energy_filter(evtpath, Emin, Emax)
    bold(evtpath)
    #get background image from event file
    bkgpath=apply_backgrounds(backgrounds, evtpath)
    return evtpath, bkgpath


def merge_events(evt_paths, obj_name, bin_pixel=8, Emin=300, Emax=12000,
                 obj_dir=home+'/objects/', update_weight=True):
    #take list of clean evt files, produce exposure map and counts image
    #make weights file if none exists
    weightpath=obj_dir+obj_name+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+'.wts'
    if not path.isdir(weightpath) or update_weights:
        weightpath=make_weigths(obj_name,Emin, Emax)
    objroot=obj_dir+obj_name+'/'
    punlearn('merge_obs')
    
    cmd=['merge_obs', ','.join(evt_paths),"outroot="+objroot, 'binsize='+str(bin_pixel),
         "bands='"+weightpath+"'",'clobber=yes', "expmapthresh='2%'"]
    shell(cmd)
    imgpath=objroot+'band1_thresh.img'
    exppath=objroot+'band1_thresh.expmap'
    fluxpath=objroot+'band1_flux.img'
    return imgpath, exppath, fluxpath


def combine_backgrounds(bkg_paths, obj_name, imgpath, obj_dir=home+'/objects/'):
    bkgpath=','.join(bkg_paths)
    outdir=obj_dir+obj_name+'/'
    outpath=outdir+'band1_thresh.bkg'
    cmd=['reproject_image', bkgpath, 'matchfile='+imgpath, 'outfile='+outpath, 'clobber=yes']
    shell(cmd)
    for b in bkg_paths:
        copypath=outdir+path.basename(b).split('_')[0]+'_band1_thresh.bkg'
        dmcopy(b, copypath)
    return outpath


#End-all function
def multi_obsid_image(obj_name, obsids=None, repro=False, update_weights=False,
                      Emin=300, Emax=12000, bin_time=200, bin_pixel=8, bad_chip=8, 
                      obs_dir=home+'/chandra_obs/', obj_dir=home+'/objects/', display=True):
    #collect parameters for cluster (Temperature and nH), make weight files
    weightpath=obj_dir+obj_name+'_'+str(Emin/1e3)+'_'+str(Emax/1e3)+'.wts'
    if not path.isdir(weightpath) or update_weights:
        weightpath=make_weigths(obj_name,Emin, Emax)
    #Collect evtfile paths and bkg file paths in these dicts
    evt_paths={}
    bkg_paths={}
    if obsids==None:
        raise NotImplementedError('Obs ID Fetch Missing')
    for o in obsids:
        #loop through obsids, reprocessing
        o=str(o)
        print('Processing ObsID', o)
        evt_paths[o], bkg_paths[o]=process_obsid(o, obj_name, repro, update_weights,
                                                             obs_dir, obj_dir, Emin, Emax,
                                                             bin_time, bin_pixel, bad_chip)
    #convert dicts into lists
    evt_paths=list(evt_paths.values())
    bkg_paths=list(bkg_paths.values())
    #get images, exposure maps from event files, merge
    img, exp, flux=merge_events(evt_paths, obj_name, bin_pixel, Emin, Emax, obj_dir, update_weights)
    #reproject background files
    bkg=combine_backgrounds(bkg_paths, obj_name, img, obj_dir)
    if display==True:
        #opens Ds9 if asked to
        ds9([img, exp, flux, bkg])
    return img, exp, flux, bkg