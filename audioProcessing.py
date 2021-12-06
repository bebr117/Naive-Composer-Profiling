# -*- coding: utf-8 -*-
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import hashlib
import time
from math import floor, ceil
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def sha256(fname):
    # taken from stack overflow, generates sha256 hash of a file from filename
    hash_sha256 = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def secsToTimestamp(secs):
    return time.strftime("%H:%M:%S", time.gmtime(secs))

def spectrogramStats(in_filename, out_folder, in_folder = "", path="", interval_len = 5, spec_fps = 20):
    '''
    Calculates some basic stats about the spectrogram of the given audio file,
    and creates a human-readable csv output file, as well as a pickle dump for
    easy use in python.
    
    Parameters
    ----------
    in_filename : String
        The filename of the input file, including extension.
    out_folder: String
        The folder to put the output files in.
    path: String
        The folder path to the input file and output file.
    interval_len: int
        Length of the intervals for the average spectrograms over time, in 
        seconds.
    spec_fps: int
        Number of frames per second for calculating the spectrogram.

    Returns
    -------
    The same dictionary that is dumped to the pickle output file.
    '''
    [Fs,x] = audioBasicIO.read_audio_file(path+in_folder+"\\"+in_filename)
    output = ""
    jsonoutput = dict() # oops at one point i used json instead of pickle and now there are relics of that everywhere haha.
    
    output += ("Filename,"+in_filename+"\n")
    jsonoutput["Filename"] = in_filename
    
    output += ("SHA256 hash,"+sha256(path+in_folder+"\\"+in_filename)+"\n")
    jsonoutput["SHA256 hash"] = sha256(path+in_folder+"\\"+in_filename)
    
    track_length = len(x)/Fs
    output += ("Length,"+secsToTimestamp(track_length)+"\n")
    jsonoutput["Length"] = track_length
    
    output += "\n"
    
    # l,r = x[:, 0], x[:, 1]
    m = audioBasicIO.stereo_to_mono(x)
    spec, t_axis, f_axis = ShortTermFeatures.spectrogram(m, Fs, (Fs/spec_fps), (Fs/(2*spec_fps)))
    
    output += "Average spectrograms over time:\n"
    output += ",Frequency axis:"
    for i in f_axis:
        output += (","+"{:.4f}".format(i))
    output += "\n"
    
    max_step = ceil(track_length/interval_len)
    fragments = [[] for _ in range(max_step)]
    for i in range(len(t_axis)):
        fragments[floor(t_axis[i]/interval_len)].append(i)
    
    i = 0
    joutarray = []
    tarray = []
    for i in range(max_step):
        if i+1 < max_step:
            output += str(i)+","+secsToTimestamp(i*interval_len)+"-"+secsToTimestamp((i+1)*interval_len)
            tarray.append(secsToTimestamp(i*interval_len)+"-"+secsToTimestamp((i+1)*interval_len))
        else:
            output += str(i)+","+secsToTimestamp(i*interval_len)+"-"+secsToTimestamp(track_length)
            tarray.append(secsToTimestamp(i*interval_len)+"-"+secsToTimestamp(track_length))
        partial_spec = spec[fragments[i][0]:fragments[i][-1]]
        partial_means = np.mean(partial_spec,axis=0)
        joutarray.append(partial_means)
        for j in partial_means:
            output+=(","+"{:.6f}".format(j))
        output += "\n"
    
    jsonoutput["Average spectrograms over time"] = (joutarray,t_axis,f_axis)
    output+="\n"
    
    output+="Overall spectrogram statistics:\n"
    joutdict = dict()
    
    output += "Frequency axis"
    for i in f_axis:
        output += (","+"{:.4f}".format(i))
    output += "\n"
    joutdict["Frequency axis"] = f_axis

    output+="Means"
    total_means = np.mean(spec,axis=0)
    for i in total_means:
        output+=(","+"{:.6f}".format(i))
    output+="\n"
    joutdict["Means"] = total_means
    
    output+="Standard deviations"
    total_stds = np.std(spec,axis=0)
    for i in total_stds:
        output+=(","+"{:.6f}".format(i))
    output+="\n"
    joutdict["Standard deviations"] = total_stds
    
    jsonoutput["Overall spectrogram statistics"] = joutdict
    
    truncated_in_filename = in_filename[:in_filename.rfind(".")]
    with open(path+out_folder+"\\"+truncated_in_filename+"stats.csv","w") as f:
        f.write(output)
    with open(path+out_folder+"\\"+truncated_in_filename+"statsPickle.p","wb") as f:
        pickle.dump(jsonoutput,f)
    
    return jsonoutput

def composerEQProfile(in_folder,out_folder,out_filename=None,path="",interval_len=5,spec_fps=20):
    '''
    Runs spectrogram analysis on a folder of .wav audio files and outputs the 
    results of those analyses and an image containing a plot of the average 
    spectrogram over all of the songs. Ideally, this average spectrogram 
    represents something about the EQ preferences of the arranger and masterer 
    of the relevant audio files.

    Parameters
    ----------
    in_folder : String
        Name of the folder with the audio files to be analyzed.
    out_folder : String
        Name of the folder to put the output files in.
    out_filename : String, optional
        Name of the output image. If falsy, the name is "[out_folder].png".
   path: String
       The folder path to the input file and output file.
   interval_len: int
       Length of the intervals for the average spectrograms over time, in 
       seconds.
   spec_fps: int
       Number of frames per second for calculating the spectrogram.

    Returns
    -------
    None.

    '''
    if not bool(out_filename):
        out_filename = out_folder+".png"
    
    files = []
    with os.scandir(path+in_folder) as in_dir:
        for f in in_dir:
            if f.name.endswith(".wav") and f.is_file():
                files.append(f.name)
    
    analyses = []
    for fname in files:
        analyses.append(spectrogramStats(fname,out_folder,in_folder=in_folder,path=path,interval_len=interval_len,spec_fps=spec_fps))
    
    # i actually DON'T want to weight by length.
    means = np.array([a["Overall spectrogram statistics"]["Means"] for a in analyses])
    overallMean = np.mean(means,axis=0)
    
    plt.plot(analyses[0]["Overall spectrogram statistics"]["Frequency axis"],overallMean,"b-")
    plt.xlabel("Frequency")
    plt.title("Average Spectrogram")
    
    plt.savefig(out_filename)