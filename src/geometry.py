def points_angle(a, o, b):
    """ Angle aob in radians. Points must be np.arrays.  """
    oa = a-o
    ob = b-o
    angle = atan2(np.cross(oa,ob), np.dot(oa,ob))
    return angle

def positive_angle(angle):
    """ Transform -2pi < angle < 2pi to 0 < angle < 2pi. """
    if angle<0:
        return angle + 2*pi
    return angle

def rotate_vector(v, radians):
    """ Roatates vector v (np.array) a given angle. """
    c, s = np.cos(radians), np.sin(radians)
    return np.dot(np.matrix([[c, s], [-s, c]]), v).A1

def points_angle_with_broadcast_2(aM, o, bM):
    """ Returns a vector with bM.shape with points_angle_with(a,o,b) for b in bM """
    oa = aM-o
    ob = bM-o
    c = np.cross(oa,ob)
    d = (ob*oa).sum(axis=1)
    angle = np.arctan2(c, d)
    return angle

def points_angle_with_broadcast_1(a, o, bM):
    """ Returns a vector with bM.shape with points_angle_with(a,o,b) for b in bM """
    oa = a-o
    ob = bM-o
    c = np.cross(oa,ob)
    d = np.dot(ob,oa)
    angle = np.arctan2(c, d)
    return angle