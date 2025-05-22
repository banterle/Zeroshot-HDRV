#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import sys
import subprocess

#
#
#
def getColorSpaceInfo(fn):
    exec_str = 'ffprobe -v error -select_streams v:0 -show_entries stream=color_space -of default=noprint_wrappers=1 \"' + fn + '\"'

    p = subprocess.Popen(exec_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_out, stderr_out = p.communicate()
    rc = p.poll()
    subprocess.call(exec_str, shell=True)
    return str(stdout_out)

#
#
#
def process1Video(path_video, fmt = 'png', bRescale = True):

    name_base = os.path.splitext(os.path.basename(path_video))[0]
    path_local = os.path.dirname(path_video)
    
    if path_local == "":
        outpath = name_base
    else:
        outpath = path_local + '/' + name_base
    
    if os.path.isdir(outpath) == False:
        os.mkdir(outpath)
    
    frame_out_str = outpath + '/' + name_base + '_%06d.' + fmt
    
    result = getColorSpaceInfo(path_video)

    if 'unknown' in result:
        exec_str = 'ffmpeg -i ' + path_video + ' ' + frame_out_str

    if ('bt709' in result):
        exec_str = 'ffmpeg -i ' + path_video + ' -vf "zscale=t=709:p=bt709:m=bt709,format=rgb24" ' + frame_out_str
        
    if ('bt470bg' in result):
        exec_str = 'ffmpeg -i ' + path_video + ' ' + frame_out_str

    if 'bt2020' in result:
        exec_str = 'ffmpeg -i ' + path_video + ' -vf "zscale=t=linear:npl=50,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=linear:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" ' + frame_out_str

    subprocess.call(exec_str, shell=True)
 
if __name__ == "__main__":

    fmt = 'png'
    folder = sys.argv[1]

    ext = os.path.splitext(folder)[1]
    ext = ext.lower()
    
    if ext  == '.mov' or ext == '.mp4':
        process1Video(folder, fmt)
    else:
        videos = [v for v in os.listdir(folder) if (v.lower().endswith('.mov') or v.lower().endswith('.mp4'))]
        for v in videos:
            print(v)
            process1Video(os.path.join(folder, v), fmt)
