"""
Processing pipeline for extracting and classifying signatures from electoral telegrams.

This module provides a complete pipeline for processing electoral telegram PDFs:
1. Convert PDFs to PNG images
2. Straighten and crop images using QR codes
3. Extract signature regions
4. Filter and classify signatures

The QR codes in telegram corners are used to detect orientation and crop boundaries.
"""

from src.all_deps import *
from src.telegram_tools import *

def telegramas_pdf_generator(PATH):
    """
    Generate filepaths for all PDF telegrams in a directory structure.
    
    Expects a directory structure where subdirectories are numeric and contain PDF files.
    
    Args:
        PATH (str): Root path containing telegram subdirectories.
    
    Yields:
        str: Full filepath to each PDF file found.
    
    Raises:
        AssertionError: If directory structure doesn't match expected format.
    """
    assert os.path.isdir(PATH)
    for subfolder in os.listdir(PATH):
        if subfolder.startswith('.'):
            continue  # skip hidden entries like .DS_Store
        subpath = os.path.join(PATH, subfolder)
        assert subfolder.isdigit()
        assert os.path.isdir(subpath)
        for pdf_filename in os.listdir(subpath):
            if pdf_filename.startswith('.'):
                continue
            assert pdf_filename.endswith('.pdf')
            pdf_filepath = os.path.join(subpath, pdf_filename)
            yield pdf_filepath

def apply_convert_from_path(ctx_output_dir, pdf_filepath):
    """
    Convert a PDF file to PNG images (one per page).
    
    Saves each page as a PNG file with naming format: PDF-{filename}-PAG-{page_num}.png
    
    Args:
        ctx_output_dir (str): Directory where PNG files will be saved.
        pdf_filepath (str): Path to the PDF file to convert.
    
    Returns:
        list: List of filepaths for all generated PNG files.
    """
    pdf_filename = os.path.basename(pdf_filepath)
    images = convert_from_path(pdf_filepath)
    result = []
    for image_idx in range(len(images)):
        pdf_identifier = 'PDF-{}-PAG-{}.png'.format(pdf_filename[:-4], image_idx)
        img = images[image_idx]
        png_filepath=os.path.join(ctx_output_dir,pdf_identifier)
        result.append(png_filepath)
        img.save(png_filepath, 'PNG')
    return result

def extract_qr_coordinates(img):
    """
    Extract coordinates from the two QR codes in a telegram image.
    
    Electoral telegrams have QR codes in opposite corners for orientation detection
    and geometric correction. This function decodes both QR codes and returns their
    polygon coordinates in the correct order.
    
    Args:
        img: PIL Image object containing the telegram.
    
    Returns:
        list: List of two 4-point polygons (one per QR code), where each point
              is a numpy array [x, y]. QR codes are ordered by their identifier.
    
    Raises:
        Exception: If exactly 2 QR codes are not detected.
    """
    dd = decode(img)
    dd = [d for d in dd if d.type=='QRCODE']
    if len(dd)!=2:
        raise Exception("Only 1 QR code detected: {}".format(dd))
    if dd[0].data[7]==48:
        assert dd[1].data[7]==49
    else:
        assert dd[1].data[7]==48
        dd = [ dd[1], dd[0] ]
    return [ [ np.array([p.x, p.y]) for p in d.polygon] for d in dd ]


def index_most_far(pts):
    """
    Find the indices of the farthest points between two QR code polygons.
    
    Args:
        pts (list): List of two 4-point polygons.
    
    Returns:
        tuple: (fi, fj) where fi is the index in first polygon and fj is the
               index in second polygon of the two farthest points.
    """
    fi, fj = -1, -1
    maxdij = 0
    for i in range(4):
        for j in range(4):
            dij = np.linalg.norm(pts[0][i]-pts[1][j])
            if dij>maxdij:
                fi = i
                fj = j
                maxdij = dij
    return fi,fj

def non_extreme_points(pts, idx):
    """
    Get adjacent and opposite points from a 4-point polygon given an index.
    
    Args:
        pts (list): List of 4 points forming a polygon.
        idx (int): Index of reference point (0-3).
    
    Returns:
        tuple: (adjacent_points, opposite_point) where adjacent_points is a list
               of the two points adjacent to idx, and opposite_point is the point
               at position (idx+2)%4.
    """
    return [ pts[i] for i in range(4) if abs(i-idx) in [1,3] ], pts[ (idx+2)%4 ]

def ordered_points(pts):
    """
    Order QR code polygon points consistently for geometric transformations.
    
    For each QR code polygon, orders the 4 corner points such that:
    - First point is the one farthest from the other QR
    - Remaining points follow clockwise order
    This consistent ordering enables accurate rotation angle calculation.
    
    Args:
        pts (list): List of two 4-point QR code polygons.
    
    Returns:
        tuple: (pts_arriba, pts_abajo) - ordered points for top and bottom QR codes.
    """
    fi, fj = index_most_far(pts)
    diag = pts[1][fj]-pts[0][fi]
    adj_up, op_up = non_extreme_points(pts[0], fi)
    adj_down, op_down = non_extreme_points(pts[1], fj)
    u = adj_up[0]-pts[0][fi]
    u = u/np.linalg.norm(u)
    v = adj_up[1]-pts[0][fi]
    v = v/np.linalg.norm(v)
    udiag = u*np.dot(diag,u)
    vdiag =  v*np.dot(diag,v)
    if np.linalg.norm(udiag)>np.linalg.norm(vdiag):
        u,v = v,u
        adj_up[0],adj_up[1] = adj_up[1],adj_up[0]
    u_down =  adj_down[0] - pts[1][fj]
    u_down = u_down/np.linalg.norm(u_down)
    v_down = adj_down[1] - pts[1][fj]
    v_down = v_down/np.linalg.norm(v_down)

    if abs(np.dot(u,v_down))+abs(np.dot(v,u_down))  > abs(np.dot(u,u_down))+abs(np.dot(v,v_down)):
        u_down,v_down = v_down,u_down
        adj_down[0],adj_down[1] = adj_down[1],adj_down[0]
    pts_arriba = [ pts[0][fi], adj_up[0], op_up, adj_up[1] ]
    pts_abajo = [op_down, adj_down[1] ,pts[1][fj] , adj_down[0]]
    return pts_arriba, pts_abajo

def get_rotation_angle_mix(pts):
    """
    Calculate multiple rotation angles from a QR code polygon for image straightening.
    
    Computes 4 different angle measurements from the QR polygon edges and diagonals
    to robustly estimate the rotation needed to straighten the image.
    
    Args:
        pts (list): 4-point polygon of a QR code.
    
    Returns:
        list: List of 4 rotation angles in radians.
    """
    def _aux_ang(i,j):
        v = pts[j]-pts[i]
        return atan2(v[1], v[0])
    return [ -_aux_ang(0,1), -_aux_ang(3,2), -_aux_ang(3,0)-pi/2, -_aux_ang(2,1)-pi/2 ]

def angle_average(angles):
    """
    Compute circular mean of angles to avoid discontinuity at 0/360 degrees.
    
    Standard arithmetic mean fails for angles near 0/2π boundary. This function
    properly handles circular averaging using trigonometric identities.
    
    Args:
        angles (list): List of angles in radians.
    
    Returns:
        float: Average angle in degrees.
    """
    return atan2(np.average(np.sin(angles)), np.average(np.cos(angles)))*180/pi

def straighten_image(img):
    """
    Straighten a telegram image using QR codes for rotation correction.
    
    Detects the two QR codes in the image, calculates the average rotation angle
    from multiple measurements, and rotates the image to correct orientation.
    This is essential for accurate signature cropping.
    
    Args:
        img: PIL Image object of the telegram.
    
    Returns:
        PIL.Image: Straightened image with expanded canvas to avoid cropping.
    """
    pts = extract_qr_coordinates(img)
    pts_arriba, pts_abajo = ordered_points(pts)
    ang = angle_average( get_rotation_angle_mix(pts_arriba) + get_rotation_angle_mix(pts_abajo) )
    return img.rotate(-ang,expand=True)

def crop_by_qr(img):
    """
    Crop telegram image to the rectangular area defined by QR code corners.
    
    Uses the farthest points of the two QR codes to define the crop boundaries,
    removing margins and focusing on the useful telegram content.
    
    Args:
        img: PIL Image object of the telegram.
    
    Returns:
        PIL.Image: Cropped image containing only the area between QR codes.
    """
    pts = extract_qr_coordinates(img)
    fi, fj = index_most_far(pts)
    return img.crop((pts[0][fi][0], pts[0][fi][1], pts[1][fj][0], pts[1][fj][1]))

def scale_image(img):
    """
    Resize telegram image to standard dimensions for consistent signature extraction.
    
    Standardizes all telegram images to 1500x2465 pixels to ensure signature
    crop coordinates work consistently across all telegrams.
    
    Args:
        img: PIL Image object.
    
    Returns:
        PIL.Image: Resized image at 1500x2465 pixels using antialiasing.
    """
    # PIL.Image.ANTIALIAS fue removido en Pillow 10; LANCZOS es el reemplazo directo.
    return img.resize((1500, 2465), PIL.Image.Resampling.LANCZOS)

def preprocess_pipeline(img):
    """
    Complete preprocessing pipeline: straighten, crop, and scale telegram image.
    
    Applies the full sequence of geometric corrections using QR codes:
    1. Straighten (correct rotation)
    2. Crop (remove margins)
    3. Scale (standardize size)
    
    Args:
        img: PIL Image object of raw telegram scan.
    
    Returns:
        PIL.Image: Preprocessed telegram ready for signature extraction.
    """
    img = straighten_image(img)
    img = crop_by_qr(img)
    img = scale_image(img)
    return img


def apply_preprocess_pipeline(ok_output_dir, error_output_dir, image_fp, delete_origin=True):
    """
    Apply preprocessing pipeline with error handling and output management.
    
    Attempts to preprocess a telegram image. Successfully processed images are saved
    to ok_output_dir, while images that fail processing (e.g., QR detection issues)
    are saved unmodified to error_output_dir for manual review.
    
    Args:
        ok_output_dir (str): Directory for successfully preprocessed images.
        error_output_dir (str): Directory for images that failed preprocessing.
        image_fp (str): Filepath of image to process.
        delete_origin (bool): Whether to delete original after processing. Defaults to True.
    
    Returns:
        str: Filepath of the output image (in either ok or error directory).
    """
    img = Image.open(image_fp)
    image_fn = os.path.basename(image_fp)
    try:
        img = preprocess_pipeline(img)
        output_fp =os.path.join(ok_output_dir,image_fn)
    except Exception as e:
        output_fp = os.path.join(error_output_dir,image_fn)
    img.save(output_fp,'PNG')
    if delete_origin:
        os.remove(image_fp)
    return output_fp

# Definimos lo 9 lugares en la hoja con firmas.
# Nombre de lugar de firma.
SIGNATURES_NAMES = ['presidente', 'aux1', 'aux2', 'fis1', 'fis2', 'fis3', 'fis4', 'jefe', 'mesa']
# Intervalos en el eje x de cada lugar de firma.
SIGNATURES_X = [(90,500),    (570,985),   (1060,1480), (90,640),    (865,1420),  (90,640),    (865,1420), (34,420), (430,860) ]
# Intervalos en el eje y de cada lugar de firma.
SIGNATURES_Y = [(1535,1670), (1535,1670), (1535,1670), (1843,1950), (1840,1935), (2051,2172),  (2051,2172), (2312, 2460), (2312, 2460)]
assert len(SIGNATURES_NAMES)==len(SIGNATURES_X)
assert len(SIGNATURES_NAMES)==len(SIGNATURES_Y)

def crop_signatures(img, margin=0):
    """
    Extract all signature regions from a preprocessed telegram page.
    
    Crops 9 predefined signature locations (president, auxiliaries, fiscales, etc.)
    from a standardized telegram image. Coordinates are fixed based on official
    telegram format used in Santa Fe elections.
    
    Args:
        img: PIL Image of preprocessed telegram page (1500x2465 pixels).
        margin (int): Extra pixels to include around signature boundaries. Defaults to 0.
    
    Returns:
        zip: Iterator of (signature_name, signature_image) tuples for 9 signature positions.
    """
    images = []
    names = []
    for i in range(len(SIGNATURES_NAMES)):
        x0,x1 = SIGNATURES_X[i]
        y0,y1 = SIGNATURES_Y[i]
        signature_img = img.crop((x0-margin,y0-margin,x1+margin,y1+margin))
        signature_img = Image.fromarray( np.array(signature_img)[:,:,0] )

        images.append(signature_img)
        names.append(SIGNATURES_NAMES[i])
    return zip(names,images)

def apply_crop_signatures(output_dir, image_fp, delete_origin=True, margin=0):
    """
    Extract and save all signature crops from a telegram page.
    
    Loads a preprocessed telegram image, extracts all 9 signature regions,
    and saves each as a separate PNG file with descriptive naming.
    
    Args:
        output_dir (str): Directory where signature crops will be saved.
        image_fp (str): Filepath of preprocessed telegram image.
        delete_origin (bool): Whether to delete source image after cropping. Defaults to True.
        margin (int): Extra pixels around signature boundaries. Defaults to 0.
    
    Returns:
        list: List of filepaths for all saved signature images.
    """
def apply_crop_signatures(output_dir, image_fp, delete_origin=True, margin=0):
    img = Image.open(image_fp)
    image_fn = os.path.basename(image_fp)
    cropped_signatures = crop_signatures(img, margin)
    result = []
    for sgn_name, sgn_img in cropped_signatures:
        output_fp = os.path.join(output_dir, image_fn[:-4] + '-' + sgn_name + '.PNG')
        sgn_img.save(output_fp,'PNG')
        result.append(output_fp)
    if delete_origin:
        os.remove(image_fp)
    return result

def negate_binarize(np_img):
    """
    Binarize image using Otsu's method and invert colors.
    
    Applies Gaussian blur followed by Otsu's automatic thresholding, then inverts
    the result so signatures appear as white pixels on black background (standard
    for signature processing).
    
    Args:
        np_img: Grayscale numpy array of image.
    
    Returns:
        numpy.ndarray: Binarized and inverted image (255 for signature, 0 for background).
    """
    _,np_img = cv2.threshold(cv2.GaussianBlur(np.array(np_img),(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return 255-np_img

def filter_signature(np_img, SOLIDITY_THRESOLD = 0.5):
    """
    Remove printed text by filtering connected components based on solidity.
    
    Separates handwritten signatures from printed characters by analyzing shape
    properties. Printed characters tend to be more solid (solidity > 0.5), while
    handwritten strokes are typically less compact. This helps isolate genuine
    signatures from form fields and printed labels.
    
    Solidity = region_area / convex_hull_area. Higher values indicate more
    filled/compact shapes typical of printed text.
    
    Args:
        np_img: Binarized numpy array where background = 0, foreground = 255.
        SOLIDITY_THRESOLD (float): Solidity threshold for filtering. Defaults to 0.5.
    
    Returns:
        numpy.ndarray: Filtered image with only low-solidity components (signatures).
    """
    blobs_labels = measure.label(np_img, background=0)
    image_label_overlay = label2rgb(blobs_labels, image=np_img)

    preserved_region = []
    for region in regionprops(blobs_labels):
        if region.solidity<SOLIDITY_THRESOLD:
            preserved_region.append(region.label)

    filtered_labels = blobs_labels.copy()
    filtered_labels[~np.isin(filtered_labels, preserved_region)] = 0

    filtered_np_img = np_img.copy()
    filtered_np_img[filtered_labels==0] = 0
    return filtered_np_img

def naive_preprocess_signature(np_img, *args, **kwargs):
    """
    Complete preprocessing for signature images: binarize and remove printed text.
    
    Two-stage pipeline:
    1. Binarize using Otsu's method
    2. Filter out printed characters using solidity-based component analysis
    
    Args:
        np_img: Grayscale numpy array of signature crop.
        *args, **kwargs: Additional parameters passed to filter_signature().
    
    Returns:
        numpy.ndarray: Clean binarized signature with printed text removed.
    """
    np_img = negate_binarize(np_img)
    np_img = filter_signature(np_img, *args, **kwargs)
    return np_img

def naive_is_signature(img, SUM_THRESOLD = 100000):
    """
    Detect presence of signature by counting white pixels.
    
    Simple heuristic classifier: if total pixel intensity exceeds threshold,
    assumes a signature is present. Works well for binary images where signatures
    appear as white strokes on black background.
    
    Args:
        img: PIL Image or numpy array of preprocessed signature crop.
        SUM_THRESOLD (int): Minimum pixel sum to classify as signature. Defaults to 100000.
    
    Returns:
        bool: True if signature detected, False otherwise.
    """
    return np.array(img).sum() >= SUM_THRESOLD

def apply_cls_signatures(is_signature_dir, is_not_signature_dir, signature_fp):
    """
    Classify a signature crop as containing or not containing a signature.
    
    Preprocesses the signature image and applies naive classification heuristic,
    then copies the file to appropriate directory based on result.
    
    Args:
        is_signature_dir (str): Directory for images classified as containing signatures.
        is_not_signature_dir (str): Directory for images classified as empty/no signature.
        signature_fp (str): Filepath of signature crop to classify.
    
    Returns:
        str: Final filepath of classified signature image.
    """
def apply_cls_signatures(is_signature_dir, is_not_signature_dir, signature_fp):
    img = Image.open(signature_fp)
    proc_image = Image.fromarray(naive_preprocess_signature(np.array(img)))
    output_dir = is_signature_dir if naive_is_signature(proc_image) else is_not_signature_dir
    signature_final_fp = os.path.join(output_dir, os.path.basename(signature_fp))
    copyfile(signature_fp, signature_final_fp)
    return signature_final_fp

if __name__ == "__main__":
    # Telegramas originales (pdfs). Tiene 2 niveles de directorio. Nombres tienen pinta '5001-51-0.pdf'
    TELEGRAMA_DATA_PATH = './telegramas-urn-sett'
    # Telegramas como PNG (un archivo por hoja). Nombres tienen pinta 'PDF-5001-51-0-PAG-0.png'
    TELEGRAMA_IMG_RAW_PATH = './telegramas-raw-data'

    # PNG preprocesados
    TELEGRAMA_IMG_CROP_PATH = './telegramas-crop-imgs'
    # PNG que dieron error al preprocesar.
    TELEGRAMA_IMG_CROP_ERROR_PATH = './telegramas-crop-imgs-error'

    # PNG firmas recortadas
    TELEGRAMA_SIGNATURES_PATH = './signatures'
    TELEGRAMA_SIGNATURES_PATH_YES = './cls_tmp/yes-signature'
    TELEGRAMA_SIGNATURES_PATH_NO = './cls_tmp/no-signature'

    if len(sys.argv)<2:
        print('Debe elegir una opcion: raw qr crop cls')
        exit(0)

    if len(sys.argv)>1 and sys.argv[1]=='raw':
        print('Generating raw pngs...')
        telegramas_paths = list(telegramas_pdf_generator(TELEGRAMA_DATA_PATH))
        # Adhocfilter
        telegramas_paths = [fp for fp in telegramas_paths if os.path.basename(fp)[0]!='3']
        if len(sys.argv)>2:
            telegramas_paths = telegramas_paths[:int(sys.argv[2])]
        print("Lodaded {} pdf's.".format(len(telegramas_paths)))
        f_map = ctx_output_dir(apply_convert_from_path, TELEGRAMA_IMG_RAW_PATH)
        imgs_fp = parallel_fmap(f_map,telegramas_paths)
        imgs_fp = flatten(imgs_fp)
        print("Generated {} png's.".format(len(imgs_fp)))

    if len(sys.argv)>1 and sys.argv[1]=='qr':
        print('Straight/Rotate pngs...')
        raw_imgs_fp = sorted([os.path.join(TELEGRAMA_IMG_RAW_PATH,fn) for fn in os.listdir(TELEGRAMA_IMG_RAW_PATH)])
        print("Lodaded {} raw png's.".format(len(raw_imgs_fp)))
        f_map = apply_preprocess_pipeline
        f_map = ctx_output_dir(f_map, TELEGRAMA_IMG_CROP_PATH)
        f_map = ctx_output_dir(f_map, TELEGRAMA_IMG_CROP_ERROR_PATH)
        f_map = functools.partial(f_map, delete_origin=True)
        result_paths = parallel_fmap(f_map, raw_imgs_fp)
        print("Processed {} png's.".format(len(result_paths)))

    if len(sys.argv)>1 and sys.argv[1]=='crop':
        margin = int(sys.argv[2]) if len(sys.argv)>=3 else 0
        print('Cropping signatures using margin={}...'.format(int(margin)))
        preproc_imgs_fp = sorted([os.path.join(TELEGRAMA_IMG_CROP_PATH,fn) for fn in os.listdir(TELEGRAMA_IMG_CROP_PATH)])
        print("Lodaded {} preprocessed png's.".format(len(preproc_imgs_fp)))
        f_map = ctx_output_dir(apply_crop_signatures, TELEGRAMA_SIGNATURES_PATH)
        f_map = functools.partial(f_map, delete_origin=False, margin=margin)
        signature_paths = parallel_fmap(f_map, preproc_imgs_fp)
        signature_paths = flatten(signature_paths)
        print("Processed {} png's.".format(len(signature_paths)))

    if len(sys.argv)>1 and sys.argv[1]=='cls':
        print('Classifying signatures...')
        signature_paths = sorted([os.path.join(TELEGRAMA_SIGNATURES_PATH,fn) for fn in os.listdir(TELEGRAMA_SIGNATURES_PATH)])
        print("Lodaded {} signatures".format(len(signature_paths)))
        f_map = apply_cls_signatures
        f_map = ctx_output_dir(f_map, TELEGRAMA_SIGNATURES_PATH_YES)
        f_map = ctx_output_dir(f_map, TELEGRAMA_SIGNATURES_PATH_NO)
        result_paths = parallel_fmap(f_map, signature_paths)
        print("Classified {} signatures.".format(len(result_paths)))
