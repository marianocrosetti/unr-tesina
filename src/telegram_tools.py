"""
Utility functions for image processing and parallel execution.

This module provides helper functions for working with telegram images including:
- Parallel processing utilities
- Image preprocessing and filtering
- Component analysis
- Visualization helpers
"""

from src.all_deps import *

def flatten(xss):
    """
    Flatten a list of lists into a single list.
    
    Args:
        xss: List of lists (or any iterable of iterables).
    
    Returns:
        list: Flattened single-level list containing all elements.
    
    Example:
        >>> flatten([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [x for xs in xss for x in xs]

def parallel_fmap(f_map, data_list):
    """
    Apply a function to a list of items in parallel with progress tracking.
    
    Uses multiprocessing with 8 workers to speed up processing of large datasets.
    Shows a tqdm progress bar during execution.
    
    Args:
        f_map: Function to apply to each item.
        data_list: List of items to process.
    
    Returns:
        list: Results from applying f_map to each item in data_list.
    """
    p = Pool(8)
    return list(tqdm.tqdm(p.imap(f_map, data_list), total=len(data_list)))

def empty_dir(dir_fp):
    """
    Empty a directory after user confirmation, creating it if it doesn't exist.
    
    Safety feature: requires user to type 'yes' before deleting directory contents.
    If directory doesn't exist, creates it empty.
    
    Args:
        dir_fp (str): Directory path to empty.
    
    Raises:
        AssertionError: If user doesn't confirm with 'yes'.
    """
    print("Are you sure to delete {}? ['yes' for yes]".format(dir_fp))
    if input()=='yes':
        if os.path.isdir(dir_fp):
            shutil.rmtree(dir_fp)
        os.makedirs(dir_fp)
    else:
        assert False

def ctx_output_dir(f_map, output_dir):
    """
    Create a partial function with output_dir pre-filled and directory prepared.
    
    Empties the output directory (with confirmation) then returns a partially
    applied version of f_map with output_dir as first argument.
    
    Args:
        f_map: Function expecting output_dir as first parameter.
        output_dir (str): Directory path to use as output.
    
    Returns:
        function: Partial function with output_dir already specified.
    """
    empty_dir(output_dir)
    return functools.partial(f_map, output_dir)

def resize_img(img, k):
    """
    Resize image by a scaling factor.
    
    Args:
        img: PIL Image object.
        k (float): Scaling factor (e.g., 0.5 for half size, 2.0 for double size).
    
    Returns:
        PIL.Image: Resized image.
    """
    return img.resize((int(img.size[0]*k), int(img.size[1]*k)))

def plot_images_grid(images,width=1, titles=[], figsize=None):
    """
    Display a list of images in a grid layout using matplotlib.
    
    Args:
        images (list): List of image arrays or PIL Images to display.
        width (int): Number of images per row. Defaults to 1.
        titles (list): Optional list of titles for each image. Defaults to [].
        figsize (tuple): Optional figure size (width, height). Defaults to None.
    """
    if figsize:
        f = plt.figure(figsize=figsize)
    n_images = len(images)
    if titles:
        assert(len(titles)==n_images)
    for i in range(n_images):
        plt.subplot((n_images+width-1)/width,width,i+1),plt.imshow(images[i],'gray')
        if titles:
            plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

def create_if_not_exists(fn):
    """
    Ensure parent directory exists for a given file path, creating if necessary.
    
    Args:
        fn (str): File path whose parent directory should exist.
    
    Returns:
        str: The input file path unchanged.
    """
    pfn=str(Path(fn).parent)
    if not os.path.exists(pfn):
        os.makedirs(pfn)
    return fn

def edges(img,blur=(3,3)):
    """
    Detect edges in an image using Canny edge detection with adaptive thresholds.
    
    Applies Gaussian blur then Canny edge detection with automatically calculated
    thresholds based on image median intensity.
    
    Args:
        img: PIL Image object.
        blur (tuple): Gaussian blur kernel size (width, height). Defaults to (3,3).
                      Set to None to skip blurring.
    
    Returns:
        numpy.ndarray: Binary edge map where edges are white (255) on black (0).
    """
    cv2img = img.convert('L')
    #img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    np_img = np.array(img)
    if blur:
        np_img = cv2.GaussianBlur(np_img,blur,0)
    median = np.median(np_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edge_image= cv2.Canny(np_img, lower, upper)
    return edge_image

def noisify(img, p):
    """
    Add random noise to an image for data augmentation.
    
    Randomly sets pixels to black with probability (1-p) to simulate scanning
    artifacts or image degradation.
    
    Args:
        img: PIL Image object.
        p (float): Probability of keeping each pixel unchanged (0.0 to 1.0).
    
    Returns:
        PIL.Image: Noisy version of input image.
    """
    npimg=np.array(img)/255
    noise = np.where(np.random.rand(*npimg.shape)>p,-255,0)
    return Image.fromarray((np.clip(noise+npimg,0,1)*255).astype(np.uint8))

def preproc(img):
    """
    Simple preprocessing: convert to grayscale and binarize with Otsu's method.
    
    Args:
        img: PIL Image object.
    
    Returns:
        PIL.Image: Binarized grayscale image.
    """
    cv2img = img.convert('L')
    cv2img = np.array(cv2img)
    cv2img.shape
    _,cv2img = cv2.threshold(cv2.GaussianBlur(cv2img,(1,1),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return Image.fromarray(cv2img)

def components(np_img):
    """
    Analyze connected components in a binary image and extract shape properties.
    
    Identifies all connected components (blobs) and computes geometric properties
    like area, solidity, and perimeter. Filters components based on solidity and
    area thresholds to identify signature-like strokes.
    
    Args:
        np_img: Binary numpy array where foreground=255, background=0.
    
    Returns:
        list: List of component property lists, each containing:
              [area, convex_area, bbox_area, solidity, perimeter]
    """
    blobs_labels = measure.label(np_img, background=0)
    image_label_overlay = label2rgb(blobs_labels, image=np_img)

    preserved_region = []
    SOLIDITY_THRESOLD = 0.25
    AREA_THRESOLD = 1000
    rows=[]
    columns=['area', 'convex_area', 'bbox_area', 'solidity', 'perimeter']
    for region in regionprops(blobs_labels):
        row = [ getattr(region,c) for c in columns ]
        rows.append(row)
        if region.solidity<SOLIDITY_THRESOLD and region.convex_area>=AREA_THRESOLD:
            #print(region.area, region.convex_area, region.solidity, region.perimeter)
            preserved_region.append(region.label)

    filtered_labels = blobs_labels.copy()
    filtered_labels[~np.isin(filtered_labels, preserved_region)] = 0

    filtered_np_img = np_img.copy()
    filtered_np_img[filtered_labels==0] = 0
    df = pd.DataFrame(data=rows,columns=columns)
    #return Image.fromarray(filtered_np_img), df
    return rows

def preproc_2(fn):
    """
    Advanced preprocessing with sharpening for better signature detection.
    
    Pipeline:
    1. Gaussian blur
    2. Convert to grayscale
    3. Apply sharpening kernel
    4. Otsu binarization
    5. Invert colors
    
    Args:
        fn (str): Filepath of image to preprocess.
    
    Returns:
        numpy.ndarray: Preprocessed binary image ready for analysis.
    """
def preproc_2(fn):
    npimg = cv2.imread(fn)
    npimg = cv2.GaussianBlur(npimg,(5,5),0)
    npimg = np.array( Image.fromarray(npimg).convert('L') )
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    npimg = cv2.filter2D(npimg, -1, sharpen_kernel)
    _,npimg = cv2.threshold(npimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    npimg=~npimg
    return npimg
