"""
Circular Grid Feature Extraction for Offline Signature Verification (Parodi's Method).

This module implements the circular grid-based feature extraction approach proposed in:
  - Parodi & Gómez (2010): "A circular grid-based feature extraction approach for off-line signature verification"
  - Parodi & Gómez (2011): "A circular grid-based rotation invariant feature extraction approach"

The method places a circular grid centered at the signature's center of mass and divides it
into N equal angular sectors. For each sector i, three features are extracted:

  1. x_PD (Pixel Density): Number of black pixels in sector i, normalized by R².
  2. x_DGC (Distance to Gravity Center): Distance from sector i's center of mass to the
     global center of mass, normalized by R (for scale invariance).
  3. x_AGC (Angle to Gravity Center): Angle from sector i's starting edge to its center
     of mass, normalized by (2π/N).

Where R is the diameter of the signature (distance between the two farthest points in the
convex hull). The final feature vector is the concatenation of all N×3 features.

Classes:
    Grid: Original (slow) implementation using dict-based indexing.
    FastGrid: Optimized implementation using numpy broadcasting.
"""
from libraries_import_all import *

from tools import *
from geometry import *


class Grid:
    """
    Circular Grid for signature feature extraction (original implementation).
    
    Divides a binary signature image into N angular sectors around the center of mass,
    computing per-sector pixel density, distance to gravity center, and angle features.
    
    Attributes:
        N_divisions: Number of angular sectors.
        center: Global center of mass of the signature (x, y).
        idx_to_pixel_count: Dict mapping sector index -> pixel count.
        idx_to_dist: Dict mapping sector index -> distance from sector's center to global center.
        idx_to_angle: Dict mapping sector index -> angle from sector start to sector's center of mass.
        R: Signature diameter (distance between the two farthest convex hull points).
        farest: Tuple of the two farthest points in the convex hull.
    """
    def __init__(self, image_fp, N_divisions):
        self.image_fp = image_fp
        """ self.N_divisions : Integer         the PIL image.                                      """
        self.N_divisions = N_divisions
        """ self.img : PIL.Image               the PIL image.                                      """
        self.img = Image.open(image_fp)
        """ self.np_img : 2D np.array          the image respective np_array.                      """
        self.np_img = np.array(self.img)
        """ self.N/M : Integer                 the images dimension (N,M) respectively for (y,x).  """
        self.N, self.M = self.np_img.shape
        """ self.center : Point                the center mass.                                    """
        yc, xc = ndimage.measurements.center_of_mass(self.np_img)
        self.center = np.array([xc,yc])
        """ self.center : Double               the grid angle.                                     """
        self.angle = 2.0*pi/N_divisions
        """ - self.idx_to_dist: [0, N_divisions) -> [ Point ]
                                map each zone index to the list of respective Points (x,y).        """
        self.coords_to_idx = defaultdict(list)
        for y in range(self.N):
            for x in range(self.M):
                point_angle = positive_angle(points_angle(self.center + np.array([0,1]),
                                                          self.center,
                                                          np.array([x,y])))
                idx = floor(point_angle/self.angle)
                self.coords_to_idx[idx].append((x,y))
        assert(set(self.coords_to_idx)==set(range(self.N_divisions)))
        """ - self.idx_to_center: [0, N_divisions) -> Point
                                  map each zone index to the respective mass center point. """
        self.idx_to_center = dict()
        self.idx_to_pixel_count = dict()
        for idx in self.coords_to_idx:
            non_zero_pto = [ (x,y) for x,y in self.coords_to_idx[idx] if self.np_img[y][x]==255 ]
            self.idx_to_pixel_count[idx] = len(non_zero_pto)
            if non_zero_pto:
                xs, ys = zip(*non_zero_pto)
                self.idx_to_center[idx] = np.array([np.average(xs),np.average(ys)])
            else:
                self.idx_to_center[idx] = self.center
        """ - self.idx_to_dist: [0, N_divisions) -> Integer
                                map each zone index to the respective distance to mass center point. """
        self.idx_to_dist = dict()
        for idx in self.idx_to_center:
            self.idx_to_dist[idx] = np.linalg.norm(self.center-self.idx_to_center[idx])
        """ - self.idx_to_vector_end: [0, N_divisions) -> Point
            vector( self.center -> self.idx_to_vector_end[idx] ) is the initial vector of the zone idx. """
        rotated_vector = np.array([0,1])
        self.idx_to_vector_end = dict()
        for idx in range(self.N_divisions):
            self.idx_to_vector_end[idx] = self.center + rotated_vector
            rotated_vector = rotate_vector(rotated_vector,-self.angle)
        """ - self.idx_to_angle: [0, N_divisions) -> Double
                          map each zone index to the respective angle from its init to its mass center. """
        self.idx_to_angle = dict()
        for idx in self.idx_to_center:
            self.idx_to_angle[idx] = positive_angle(points_angle(self.idx_to_vector_end[idx],
                                                                 self.center,
                                                                 self.idx_to_center[idx]))
        """ - self.chull_pts : [ Point ]                   Points of the convex hull of the signature.
            - self.farest : (Point, Point)                 Two farest points.
            - self.R : Double                              Distance between self.farest.                """
        pts_nonzero = [ (x,y) for y in range(self.N) for x in range(self.M) if self.np_img[y][x]==255 ]
        # two points which are fruthest apart will occur as vertices of the convex hull
        self.chull_pts = np.array(pts_nonzero)[spatial.ConvexHull(pts_nonzero).vertices]
        # get distances between each pair of candidate points
        dist_mat = spatial.distance_matrix(self.chull_pts, self.chull_pts)
        # get indices of candidates that are furthest apart
        i,j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        self.farest = (self.chull_pts[i], self.chull_pts[j])
        self.R      =  np.linalg.norm(self.chull_pts[i] - self.chull_pts[j])

class FastGrid:
    """
    Optimized Circular Grid for signature feature extraction using numpy broadcasting.
    
    Same as Grid but uses vectorized operations for faster computation.
    
    Attributes:
        N_divisions: Number of angular sectors.
        center: Global center of mass of the signature (x, y).
        idx_to_pixel_count: Array[N_divisions] with pixel count per sector.
        idx_to_dist: Array[N_divisions] with distance from sector's center to global center.
        idx_to_angle: Array[N_divisions] with angle from sector start to sector's center of mass.
        R: Signature diameter (distance between the two farthest convex hull points).
    """
    def __init__(self, image_fp, get_x, N_divisions):
        self.image_fp = image_fp
        """ self.N_divisions : Integer         the PIL image.                                      """
        self.N_divisions = N_divisions
        """ self.img : PIL.Image               the PIL image.                                      """
        self.img = get_x(image_fp)
        """ self.npimg : 2D np.array          the image respective np_array.                      """
        self.npimg = np.array(self.img)
        """ self.N/M : Integer                 the images dimension (N,M) respectively for (y,x).  """
        self.N, self.M = self.npimg.shape
        """ self.center : Point                the center mass.                                    """
        yc, xc = ndimage.measurements.center_of_mass(self.npimg)
        self.center = np.array([xc,yc])
        """ self.center : Double               the grid angle.                                     """
        self.angle = 2.0*pi/N_divisions

        """ self.center : NxM np.array 
                where self.center[i][j] = angle from point(i,j) to center respect to the vertical. """
        x = np.array(range(self.npimg.shape[1]))
        y = np.array(range(self.npimg.shape[0]))
        x, y = np.meshgrid(x, y)
        angs = points_angle_with_broadcast_1(self.center + np.array([0,1]), self.center, np.moveaxis(np.array([x,y]), 0,-1))
        angs = np.where(angs<0,angs+2*np.pi,angs)
        self.angs = np.floor(angs/self.angle).astype(np.int32)

        """ - self.idx_to_center: N_divisionsx2 np.array
                self.idx_to_center[i] is the respective mass center point. """
        """ - self.idx_to_pixel_count: N_divisions np.array
                self.idx_to_pixel_count[i] is the number of nonzero pixeles in each region. """
        self.idx_to_center = np.zeros((N_divisions,2))
        self.idx_to_pixel_count = np.zeros(N_divisions)
        for idx in range(self.N_divisions):
            npaux=self.npimg.copy()
            npaux[self.angs!=idx]=0
            self.idx_to_pixel_count[idx] = (npaux!=0).sum()
            if npaux.any():
                x,y=ndimage.measurements.center_of_mass(npaux)
                self.idx_to_center[idx]=np.array([y,x])
            else:
                self.idx_to_center[idx]=self.center
        
        """ - self.idx_to_dist: N_divisions np.array
                self.idx_to_dist[i] is the distance to mass center point. """
        self.idx_to_dist=np.linalg.norm(self.center-self.idx_to_center,axis=1)
        
        """ - self.idx_to_vector_end: N_divisionsx2 np.array
                self.idx_to_vector_end[i] is the initial vector of the zone idx. """
        radians=np.array(range(self.N_divisions))*-self.angle
        c, s = np.cos(radians), np.sin(radians)
        v=np.array([0,1])
        self.idx_to_vector_end=np.moveaxis(np.dot(v,np.array(([np.array([c, s]), np.array([-s, c])]))),0,1)+self.center
        
        """ - self.idx_to_angle: N_divisions np.array
                self.idx_to_angle[i] is the respective angle from its init to its mass center. """
        idx_to_angle = points_angle_with_broadcast_2(self.idx_to_vector_end, self.center, self.idx_to_center)
        self.idx_to_angle = np.where(idx_to_angle<0,idx_to_angle+2.0*np.pi,idx_to_angle)
        """ - self.chull_pts : [ Point ]                   Points of the convex hull of the signature.
            - self.farest : (Point, Point)                 Two farest points.
            - self.R : Double                              Distance between self.farest.                """
        y,x=np.where(self.npimg==255)
        nz=np.moveaxis(np.array([x,y]),0,1)
        self.chull_pts = nz[spatial.ConvexHull(nz).vertices]
        dist_mat = spatial.distance_matrix(self.chull_pts, self.chull_pts)
        i,j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        self.farest = (self.chull_pts[i], self.chull_pts[j])
        self.R      =  np.linalg.norm(self.chull_pts[i] - self.chull_pts[j])
    
    def __getstate__(self):
        state = self.__dict__.copy()
        k_to_del = [ 'img', 'npimg', 'chull_pts']
        for k in k_to_del:
            if k in state:
                del state[k]
        return state

    def features(self):
        """
        Compute the normalized feature vector for this signature.
        
        Returns:
            np.array of shape (3, N_divisions) with:
              - [0]: Pixel density (x_PD), normalized by R² and scaled by N.
              - [1]: Distance to gravity center (x_DGC), normalized by R.
              - [2]: Angle to gravity center (x_AGC), scaled by N.
        """
        pixel_count_feature = self.idx_to_pixel_count * self.N_divisions / (self.R*self.R) 
        dist_feature = self.idx_to_dist / self.R
        angle_feature = self.idx_to_angle * self.N_divisions
        return np.array([pixel_count_feature, dist_feature, angle_feature])

def idx_dict_to_numpy(a):
    """Convert dict[int->value] to numpy array ordered by key."""
    return np.array([a[i] for i in range(len(a))])

def checks():
    """Validate that FastGrid produces identical results to Grid."""
    PREPROCESSED_PATH='./data/parodi_out/preproc'
    get_fns = ds_sigcomp2009.get_authentic_signatures
    get_x = ds_sigcomp2009.get_x
    preprocess_dataset = L(get_fns(PREPROCESSED_PATH))

    for f in preprocess_dataset:
        print(f)
        fgrid = FastGrid(f, ds_sigcomp2009.get_x, N_divisions=16)
        grid = Grid(f, N_divisions=16)
        assert(grid.N_divisions == fgrid.N_divisions )
        assert(grid.N == fgrid.N )
        assert(grid.M == fgrid.M )
        assert((grid.center == fgrid.center).all())
        assert( (idx_dict_to_numpy(grid.idx_to_center)==fgrid.idx_to_center).all() )
        assert( (idx_dict_to_numpy(grid.idx_to_pixel_count)==fgrid.idx_to_pixel_count).all() )
        assert( (idx_dict_to_numpy(grid.idx_to_dist)==fgrid.idx_to_dist).all() )
        assert( (idx_dict_to_numpy(grid.idx_to_vector_end)==fgrid.idx_to_vector_end).all() )
        assert( (idx_dict_to_numpy(grid.idx_to_angle)==fgrid.idx_to_angle).all() )
        assert( ( np.array(grid.farest) == np.array(fgrid.farest) ).all() )
        assert( grid.R == fgrid.R )
        print("ok")

