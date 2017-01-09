import re
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
