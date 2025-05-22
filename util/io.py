#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os

#
#
#
def getFilenameOnly(path):
    fn = os.path.basename(path)
    fn = os.path.splitext(fn)[0]
    return fn

#
#
#
def mkdir_s(output_dir):
    if os.path.isdir(output_dir) == False:
       #print('Make folder: ' + output_dir)
       os.mkdir(output_dir)
      
#
#
#
def getGlobalPath(str):
    str_i = str[::-1]
    index = str_i.find('/')
    if index == -1:
       return ''
    else:
       icut = len(str) - index
       return str[0:icut]
