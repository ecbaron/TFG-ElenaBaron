import os
import SimpleITK as sitk
import numpy as np

# Configuración
ruta_imagenes = r"C:\Users\elena\Documents\TFG\imagenes_cortes\reescaladas_y_normalizadas_definitivo"
ruta_segmentaciones = r"C:\Users\elena\Documents\TFG\segmentaciones\reescaladas_definitivo"

# Carpetas de salida
ruta_salida_imagenes = os.path.join(ruta_imagenes, "img_cut_def")
ruta_salida_mascaras = os.path.join(ruta_segmentaciones, "mask_cut_def")

os.makedirs(ruta_salida_imagenes, exist_ok=True)
os.makedirs(ruta_salida_mascaras, exist_ok=True)

# Tamaño fijo del recorte en X, Y, Z (en voxeles)
target_xyz = (448, 448, 448)

# Pacientes sin segmentación con centros definidos manualmente
centros_manuales = {
    "6041596.nii": (464, 463, 348),
    "6513454.nii": (410, 349, 253),
    "6698358.nii": (400, 350, 470)
}

def buscar_segmentacion(nombre_imagen, carpeta_segmentaciones):
    """ Busca la segmentación asociada a la imagen """
    numero = os.path.splitext(nombre_imagen)[0]
    for archivo in os.listdir(carpeta_segmentaciones):
        if numero in archivo and archivo.endswith(".nrrd"):
            return os.path.join(carpeta_segmentaciones, archivo)
    return None

def calcular_centroide(mask_array):
    """ Calcula el centroide ponderado de la estructura segmentada. """
    nonzero = np.nonzero(mask_array > 0)
    centro_z = int(np.mean(nonzero[0]))
    centro_y = int(np.mean(nonzero[1]))
    centro_x = int(np.mean(nonzero[2]))
    return centro_z, centro_y, centro_x

def recorte_desde_bbox(array, centro_z, centro_y, centro_x, target_xyz):
    """ Realiza el recorte centrado y aplica padding de forma centrada. """
    z_dim, y_dim, x_dim = array.shape
    target_z, target_y, target_x = target_xyz

    # Calcular los índices iniciales y finales en cada eje
    z_ini = max(centro_z - target_z // 2, 0)
    y_ini = max(centro_y - target_y // 2, 0)
    x_ini = max(centro_x - target_x // 2, 0)

    z_fin = z_ini + target_z
    y_fin = y_ini + target_y
    x_fin = x_ini + target_x

    # Ajustar si nos salimos del rango
    if z_fin > z_dim:
        z_ini = max(z_dim - target_z, 0)
        z_fin = z_dim

    if y_fin > y_dim:
        y_ini = max(y_dim - target_y, 0)
        y_fin = y_dim

    if x_fin > x_dim:
        x_ini = max(x_dim - target_x, 0)
        x_fin = x_dim

    # Recorte
    array_crop = array[z_ini:z_fin, y_ini:y_fin, x_ini:x_fin]

    # Calcular padding necesario
    pad_z = max(0, target_z - array_crop.shape[0])
    pad_y = max(0, target_y - array_crop.shape[1])
    pad_x = max(0, target_x - array_crop.shape[2])

    # Distribuir el padding de forma centrada
    pad_z_before = pad_z // 2
    pad_z_after = pad_z - pad_z_before

    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before

    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before

    # Aplicar padding con ceros de forma centrada
    array_crop = np.pad(array_crop, 
                        ((pad_z_before, pad_z_after), 
                         (pad_y_before, pad_y_after), 
                         (pad_x_before, pad_x_after)), 
                        mode='constant', constant_values=0)

    return array_crop

def guardar_imagen(array, referencia, ruta_salida):
    """ Guarda la imagen manteniendo la metadata del original. """
    imagen_recortada = sitk.GetImageFromArray(array)
    imagen_recortada.SetOrigin(referencia.GetOrigin())
    imagen_recortada.SetSpacing(referencia.GetSpacing())
    imagen_recortada.SetDirection(referencia.GetDirection())
    sitk.WriteImage(imagen_recortada, ruta_salida)

# Procesamiento de Imágenes y Máscaras
for archivo in os.listdir(ruta_imagenes):
    if archivo.endswith(".nii") or archivo.endswith(".nii.gz"):
        ruta_img = os.path.join(ruta_imagenes, archivo)
        ruta_seg = buscar_segmentacion(archivo, ruta_segmentaciones)
        salida_img = os.path.join(ruta_salida_imagenes, archivo)

        print(f"\n📂 Procesando imagen: {archivo}")
        imagen = sitk.ReadImage(ruta_img)

        if ruta_seg:
            print(f"   🧠 Segmentación encontrada: {os.path.basename(ruta_seg)}")
            mask = sitk.ReadImage(ruta_seg)
            mask_array = sitk.GetArrayFromImage(mask)

            if np.count_nonzero(mask_array) == 0:
                print(f"   ⚠️ La máscara está vacía. Omitiendo.")
                continue

            try:
                centro_z, centro_y, centro_x = calcular_centroide(mask_array)
                print(f"   📍 Centro calculado: (Z={centro_z}, Y={centro_y}, X={centro_x})")

                # Recortar imagen
                array_img = sitk.GetArrayFromImage(imagen)
                img_crop = recorte_desde_bbox(array_img, centro_z, centro_y, centro_x, target_xyz)
                guardar_imagen(img_crop, imagen, salida_img)
                print(f"   ✅ Guardado imagen: {salida_img}")

                # Recortar máscara
                salida_mask = os.path.join(ruta_salida_mascaras, os.path.basename(ruta_seg))
                mask_crop = recorte_desde_bbox(mask_array, centro_z, centro_y, centro_x, target_xyz)
                guardar_imagen(mask_crop, mask, salida_mask)
                print(f"   ✅ Guardado máscara: {salida_mask}")

            except Exception as e:
                print(f"   ❌ Error al procesar {archivo}: {e}")

        elif archivo in centros_manuales:
            print("   📝 Centro definido manualmente.")
            centro_x, centro_y, centro_z = centros_manuales[archivo]

            # Recortar imagen sin máscara
            array_img = sitk.GetArrayFromImage(imagen)
            img_crop = recorte_desde_bbox(array_img, centro_z, centro_y, centro_x, target_xyz)
            guardar_imagen(img_crop, imagen, salida_img)
            print(f"   ✅ Guardado imagen: {salida_img}")

        else:
            print("   ⚠️ No hay segmentación ni centro manual → omitiendo.")
