import re
import numpy as np
import pandas as pd
def get_bbox(string):

    bbox=[]
    if type(string)==type(''):
        for i in re.findall('\d+\.+',string):
            try:
                bbox.append(int(round(float(i))))
            except:
                continue
    if type(string)==type([]):
        for i in string:
            try:
                bbox.append(int(round(float(i))))
            except:
                continue
    return bbox

def get_overlap(l1,l2):
    xa1,ya1,xa2,ya2=l1[0],l1[1],l1[2],l1[3]
    xb1,yb1,xb2,yb2=l2[0],l2[1],l2[2],l2[3]
    dx = min(xa2, xb2) - max(xa1, xb1)
    dy = min(ya2, yb2) - max(ya1, yb1)
    area_a=(xa2-xa1)*(ya2-ya1)
    area_b=(xb2-xb1)*(yb2-yb1)
    if (dx>=0) and (dy>=0):
        area_i= dx*dy
        area=area_a+area_b-area_i
        return float(area_i)/area
    else:
        return 0

def merge_duplicates(df):
    drop=[]
    for i in df.index:
        for j in df[df[0]==df.loc[i][0]].index:
            if i!=j:
                l1=get_bbox(df.loc[i][1])
                l2=get_bbox(df.loc[j][1])
                if len(l1)!=4:
                    print i,l1
                if len(l2)!=4:
                    print j,l2
                if get_overlap(l1,l2)>0.6:
                    if (df.loc[i][3]==df.loc[j][3]) & (df.loc[i][2]==df.loc[j][2]):
                        drop.append(i)
                    elif (df.loc[i][3]==df.loc[j][3]) & (df.loc[i][2]!=df.loc[j][2]):
                        continue
                    elif min(df.loc[i][3],df.loc[j][3])==df.loc[i][3]:
                        drop.append(i)
                    else:
                        drop.append(j)
    return np.unique(drop)
