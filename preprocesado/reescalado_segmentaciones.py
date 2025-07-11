import os
import SimpleITK as sitk

# Ruta de entrada (máscaras originales)
directorio_entrada = r"C:\Users\elena\Documents\TFG\segmentaciones"

# Ruta de salida (máscaras reescaladas)
directorio_salida = os.path.join(directorio_entrada, "reescaladas_definitivo")
os.makedirs(directorio_salida, exist_ok=True)

# Resolución objetivo
resolucion_objetivo = [0.5, 0.5, 0.5]

# Lista de nombres de las segmentaciones (las que hiciste manualmente)
segmentaciones = ["SegPaciente1297164_FINAL.seg.nrrd", "SegPaciente1405347_FINAL.seg.nrrd", "SegPaciente1637400_FINAL.seg.nrrd", 
                  "SegPaciente1741355_FINAL.seg.nrrd", "SegPaciente2851790_FINAL.seg.nrrd", "SegPaciente6554609_FINAL.seg.nrrd"]  # o .nrrd si es el caso

for nombre in segmentaciones:
    ruta_original = os.path.join(directorio_entrada, nombre)
    ruta_salida = os.path.join(directorio_salida, nombre)

    print(f"📂 Reescalando segmentación: {nombre}...")

    img = sitk.ReadImage(ruta_original)
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
        sitk.sitkNearestNeighbor,  #Interpolación especial para máscaras
        img.GetOrigin(),
        resolucion_objetivo,
        img.GetDirection(),
        0,
        img.GetPixelID()
    )

    sitk.WriteImage(img_resampleada, ruta_salida)
    print(f"Segmentación guardada en: {ruta_salida}\n")
