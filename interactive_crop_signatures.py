#!/usr/bin/env python3
# %%
# ### Imports
# Acá importamos cosillas

# %%
import sys
print(sys.version)
sys.path.append('python-src')
sys.path.append('deteccion-firmas-master/python-src')
from import_all_notebook_hook import *

# %% [markdown]
asd

# %% [markdown]
# Generacion de dataset de firmas
# 
# Vamos a ver como agarramos un pdf y lo transformamos en una lista de firmas. Para eso:
# - `convert_from_path(pdf_path)`: nos da una lista de PNGs por hoja dado el path de un pdf.
# - `preprocess_pipeline(img)`: recorta, endereza usando QR's.
# - `crop_signatures(img)`: dada una imagen con una hoja de un telegrama devuelve una lista de (nombre firma, imagen firma).
# - `naive_preprocess_signature`: un procesamiento naive para extraer solo la firma

# %%
pdfs = convert_from_path('deteccion-firmas-master/example.pdf')
img = pdfs[0]
resize_img(img, .2)

# %%
img = preprocess_pipeline(img)
resize_img(img, .2)

# %%
nombres, imgs = list(zip(*crop_signatures(img)))
plot_images_grid(imgs, 3, nombres, figsize=(15, 5))

# %%
nombres, imgs = list(zip(*crop_signatures(img)))
proc_imgs = [Image.fromarray(naive_preprocess_signature(np.array(img))) for img in imgs]
is_sgn = ["SIGNATURE!" if naive_is_signature(img) else "NO SIGNATURE" for img in proc_imgs]
plot_images_grid(proc_imgs, 3, is_sgn, figsize=(15, 5))

# %%
# ## Procesamiento de todos los archivos
# 
# A continuación estan las funciones que leen todos los pdfs, y le aplican el pipeline anterior.
# 
# ### Definicion de PATH con resultados del pipeline 
# 
# Vamos a ir procesando (paralelizando) y guardando resultados temporales.

# %%
# Telegramas originales (pdfs). Tiene 2 niveles de directorio. Nombres tienen pinta '5001-51-0.pdf'
TELEGRAMA_DATA_PATH = './deteccion-firmas-master/telegramas-urn-sett'
# Telegramas como PNG (un archivo por hoja). Nombres tienen pinta 'PDF-5001-51-0-PAG-0.png'
TELEGRAMA_IMG_RAW_PATH = './deteccion-firmas-master/telegramas-raw-data'

# PNG preprocesados
TELEGRAMA_IMG_CROP_PATH = './deteccion-firmas-master/telegramas-crop-imgs'
# PNG que dieron error al preprocesar.
TELEGRAMA_IMG_CROP_ERROR_PATH = './deteccion-firmas-master/telegramas-crop-imgs-error'

# PNG firmas recortadas
TELEGRAMA_SIGNATURES_PATH = './deteccion-firmas-master/signatures'
TELEGRAMA_SIGNATURES_PATH_YES = './deteccion-firmas-master/cls_tmp/yes-signature'
TELEGRAMA_SIGNATURES_PATH_NO = './deteccion-firmas-master/cls_tmp/no-signature'

# %%
# ### Leemos los telegramas

# %%
telegramas_paths = list(telegramas_pdf_generator(TELEGRAMA_DATA_PATH))
print("Lodaded {} pdf's.".format(len(telegramas_paths)))
# Todos los pdf's de la carpeta ./telegramas-urn-sett/x empeizan su nombre de archivo con digito x (no hay colisiones)
assert all([os.path.basename(os.path.dirname(fp)) == os.path.basename(fp)[0] for fp in telegramas_paths])
telegramas_paths[:10]

# %%
# #### Ejecutamos lo siguiente para no trabajar con todos (para una prueba por ejemplo)

# %%
telegramas_paths = telegramas_paths[:20]

# %%
# ### Convertimos los pdfs a imagenes (una imagen por hoja)

# %%
f_map = ctx_output_dir(apply_convert_from_path, TELEGRAMA_IMG_RAW_PATH)
imgs_fp = parallel_fmap(f_map, telegramas_paths)
imgs_fp = flatten(imgs_fp)
print("Generated {} png's.".format(len(imgs_fp)))
imgs_fp[:10]

# %%
resize_img(Image.open(imgs_fp[0]), .2)

# %%
# ### Leemos los archivos png de imagen

# %%
raw_imgs_fp = sorted([os.path.join(TELEGRAMA_IMG_RAW_PATH, fn) for fn in os.listdir(TELEGRAMA_IMG_RAW_PATH)])
print("Lodaded {} raw png's.".format(len(raw_imgs_fp)))
raw_imgs_fp[:3]

# %%
# ### Preprocesamos las imagenes y guardamos las que generan error y los resultados

# %%
f_map = apply_preprocess_pipeline
f_map = ctx_output_dir(f_map, TELEGRAMA_IMG_CROP_PATH)
f_map = ctx_output_dir(f_map, TELEGRAMA_IMG_CROP_ERROR_PATH)
f_map = functools.partial(f_map, delete_origin=True)

result_paths = parallel_fmap(f_map, raw_imgs_fp)
print("Processed {} png's.".format(len(result_paths)))
result_paths[:10]

# %%
# ### Una que generó error la mostramos y vemos el error que genera
# 
# El error es que solo encuentra 1 QR. Vamos a ignorar las hojas que den errores. Muchos de estos errores no nos van a llegar (si no se detecta un QR se arrojará un error antes)

# %%
img = Image.open('./deteccion-firmas-master/telegramas-crop-imgs-error/PDF-3148-51-0-PAG-0.png')
resize_img(img, .4)

# %%
img = preprocess_pipeline(img)

# %%
# ### Leemos los PNGs preprocesados

# %%
preproc_imgs_fp = sorted([os.path.join(TELEGRAMA_IMG_CROP_PATH, fn) for fn in os.listdir(TELEGRAMA_IMG_CROP_PATH)])
print("Lodaded {} preprocessed png's.".format(len(preproc_imgs_fp)))
preproc_imgs_fp

# %%
# ### Recortamos las firmas de cada pagina de telegrama y guardamos

# %%
f_map = ctx_output_dir(apply_crop_signatures, TELEGRAMA_SIGNATURES_PATH)
f_map = functools.partial(f_map, delete_origin=False)

signature_paths = parallel_fmap(f_map, preproc_imgs_fp)
signature_paths = flatten(signature_paths)
print("Processed {} png's.".format(len(signature_paths)))
signature_paths[:10]

# %%
# ### Leemos el directorio con todas las firmas

# %%
signature_paths = sorted([os.path.join(TELEGRAMA_SIGNATURES_PATH, fn) for fn in os.listdir(TELEGRAMA_SIGNATURES_PATH)])
print("Lodaded {} signatures".format(len(signature_paths)))
signature_paths[:10]

# %%
# ### Aqui un ejemplo de firmas y su procesamiento/clasificacion naive

# %%
N_SAMPLE = 10

imgs = [Image.open(fp) for fp in random.sample(signature_paths, N_SAMPLE)]
proc_imgs = [Image.fromarray(naive_preprocess_signature(np.array(img))) for img in imgs]
cls = ["SIGNATURE!" if naive_is_signature(img) else "NO SIGNATURE" for img in proc_imgs]

images = flatten(list(zip(imgs, proc_imgs)))
titles = flatten(list(zip(cls, cls)))

plot_images_grid(images, 4, titles, figsize=(25, 15))

# %%
# ### Clasificamos las firmas y guardamos en dos directorios

# %%
f_map = apply_cls_signatures
f_map = ctx_output_dir(f_map, TELEGRAMA_SIGNATURES_PATH_YES)
f_map = ctx_output_dir(f_map, TELEGRAMA_SIGNATURES_PATH_NO)

result_paths = parallel_fmap(f_map, signature_paths)
print("Classified {} signatures.".format(len(result_paths)))
result_paths[:10]

# %%
import fastbook
fastbook.setup_book()
from fastbook import *

# %%
get_images('./cls_test')

# %%
