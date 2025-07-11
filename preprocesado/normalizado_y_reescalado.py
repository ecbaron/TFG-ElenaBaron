import os
import SimpleITK as sitk

# Rutas
directorio_entrada = r"C:\Users\elena\Documents\TFG\imagenes_cortes"
directorio_salida = os.path.join(directorio_entrada, "probando_probando")
os.makedirs(directorio_salida, exist_ok=True)

# Resolución objetivo (zoom digital)
resolucion_objetivo = [0.5, 0.5, 0.5]

# Clipping y normalización: rango para TAC torácico
CLIP_MIN = -200.0
CLIP_MAX = 300.0

# Archivos a procesar
imagenes_nombres = ["1297164.nii", "1405347.nii", "1637400.nii",  "1741355.nii",  "2851790.nii",  "6041596.nii",  "6513454.nii",  "6554609.nii",  "6698358.nii"]

for nombre in imagenes_nombres:
    ruta_original = os.path.join(directorio_entrada, nombre)
    ruta_salida = os.path.join(directorio_salida, nombre)

    print(f"Procesando {nombre}...")

    try:
        # Leer imagen
        img = sitk.ReadImage(ruta_original, sitk.sitkFloat32)  # ← cast directo a float32

        # Reescalar
        tamaño_original = img.GetSize()
        espaciado_original = img.GetSpacing()

        nuevo_tamaño = [
            int(round(tamaño_original[i] * (espaciado_original[i] / resolucion_objetivo[i])))
            for i in range(3)
        ]

        img_resampleada = sitk.Resample(
            img,
            nuevo_tamaño,
            sitk.Transform(),
            sitk.sitkLinear,
            img.GetOrigin(),
            resolucion_objetivo,
            img.GetDirection(),
            0.0,
            sitk.sitkFloat32
        )

        # Normalización sin convertir a array
        img_clip = sitk.Clamp(img_resampleada, lowerBound=CLIP_MIN, upperBound=CLIP_MAX)
        img_normalizada = (img_clip - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)

        # Guardar
        sitk.WriteImage(img_normalizada, ruta_salida)
        print(f"✅ Guardado en: {ruta_salida}\n")

    except Exception as e:
        print(f"❌ Error procesando {nombre}: {e}\n")
