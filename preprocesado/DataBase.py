import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

# Configuración
ruta_imagenes = r"C:\Users\elena\Documents\TFG\imagenes_cortes"
ruta_mascaras = r"C:\Users\elena\Documents\TFG\segmentaciones"

ruta_salida_imagenes = os.path.join(ruta_imagenes, "prueba_img")
ruta_salida_mascaras = os.path.join(ruta_imagenes, "prueba_mask")

os.makedirs(ruta_salida_imagenes, exist_ok=True)
os.makedirs(ruta_salida_mascaras, exist_ok=True)

def buscar_mascara(paciente_id, carpeta_mascaras):
    for archivo in os.listdir(carpeta_mascaras):
        if paciente_id in archivo and archivo.endswith(".nrrd"):
            return os.path.join(carpeta_mascaras, archivo)
    return None

def crear_directorio_paciente(base_path, paciente_id):
    carpeta_paciente = os.path.join(base_path, paciente_id)
    os.makedirs(carpeta_paciente, exist_ok=True)
    return carpeta_paciente

def convertir_a_binario(array):
    # Convertir a binario manteniendo las áreas segmentadas completas
    return np.where(array > 0, 1, 0)

def guardar_corte(corte_array, ruta_salida, nombre):
    # Asegurar que la máscara sea binaria completa
    corte_array = (corte_array * 255).astype(np.uint8)
    img = Image.fromarray(corte_array)
    img.save(os.path.join(ruta_salida, f"{nombre}.png"))

def procesar_paciente(ruta_img, ruta_mask, paciente_id):
    print(f"\n📂 Procesando: {paciente_id}")

    carpeta_img = crear_directorio_paciente(ruta_salida_imagenes, paciente_id)
    carpeta_mask = crear_directorio_paciente(ruta_salida_mascaras, paciente_id)

    # Leer imagen y estandarizar orientación
    imagen = sitk.ReadImage(ruta_img)
    imagen = sitk.DICOMOrient(imagen, 'LPS')  # Orientación uniforme
    array_img = sitk.GetArrayFromImage(imagen)  # Z, Y, X

    # Leer máscara si existe
    if ruta_mask:
        mascara = sitk.ReadImage(ruta_mask)
        mascara = sitk.DICOMOrient(mascara, 'LPS')  # Orientación uniforme
        array_mask = sitk.GetArrayFromImage(mascara)
        array_mask = convertir_a_binario(array_mask)
        min_z = min(array_img.shape[0], array_mask.shape[0])
        array_img = array_img[:min_z, :, :]
        array_mask = array_mask[:min_z, :, :]
    else:
        array_mask = None

    # Eje Axial (Z)
    for z in range(array_img.shape[0]):
        nombre = f"{paciente_id}_axial_{str(z).zfill(4)}"
        guardar_corte(array_img[z, :, :], carpeta_img, nombre)
        if array_mask is not None:
            guardar_corte(array_mask[z, :, :], carpeta_mask, nombre)

    # Eje Coronal (Y)
    for y in range(array_img.shape[1]):
        nombre = f"{paciente_id}_coronal_{str(y).zfill(4)}"
        corte_coronal = np.flipud(array_img[:, y, :])
        guardar_corte(corte_coronal, carpeta_img, nombre)
        if array_mask is not None:
            corte_coronal_mask = np.flipud(array_mask[:, y, :])
            guardar_corte(corte_coronal_mask, carpeta_mask, nombre)

    # Eje Sagital (X) - Ajustado a la orientación de la referencia
    for x in range(array_img.shape[2]):
        nombre = f"{paciente_id}_sagital_{str(x).zfill(4)}"
        corte_sagital = np.flipud(array_img[:, :, x])
        guardar_corte(corte_sagital, carpeta_img, nombre)
        if array_mask is not None:
            corte_sagital_mask = np.flipud(array_mask[:, :, x])
            guardar_corte(corte_sagital_mask, carpeta_mask, nombre)

# Procesar todas las imágenes
for archivo in os.listdir(ruta_imagenes):
    if archivo.endswith((".nii", ".nii.gz")):
        paciente_id = os.path.splitext(archivo)[0]
        ruta_img = os.path.join(ruta_imagenes, archivo)
        ruta_mask = buscar_mascara(paciente_id, ruta_mascaras)
        procesar_paciente(ruta_img, ruta_mask, paciente_id)
