import os
import SimpleITK as sitk
import numpy as np
import nrrd

# Ruta de la carpeta con segmentaciones
ruta_segmentaciones = r"C:\Users\elena\Documents\TFG\segmentaciones\reescaladas_definitivo\mask_cut_def"

# Ruta de salida
ruta_salida = os.path.join(ruta_segmentaciones, "prueba")
os.makedirs(ruta_salida, exist_ok=True)


def convertir_a_binario(segmentacion_path):
    """
    Convierte a binario manteniendo la estructura original.
    La etiqueta 0 será siempre fondo (0) y la etiqueta con más píxeles también se asigna como fondo (0).
    Todo lo demás se asigna a 1.
    """
    # Leer la segmentación
    data, header = nrrd.read(segmentacion_path)

    # Obtener los valores únicos y el número de píxeles de cada etiqueta
    unique, counts = np.unique(data, return_counts=True)
    etiqueta_pixel_counts = dict(zip(unique, counts))

    print(f"Valores únicos antes del procesamiento: {unique}")
    print("Número de píxeles por etiqueta:")
    for etiqueta, cantidad in etiqueta_pixel_counts.items():
        print(f"  - Etiqueta {etiqueta}: {cantidad} píxeles")

    # La etiqueta 0 es siempre fondo
    etiqueta_fondo = 0

    # Identificar la etiqueta con más píxeles (excluyendo el 0)
    etiquetas_restantes = {k: v for k, v in etiqueta_pixel_counts.items() if k != 0}
    if etiquetas_restantes:
        etiqueta_mayor = max(etiquetas_restantes, key=etiquetas_restantes.get)
        print(f"  - Etiqueta con más píxeles (excluyendo 0): {etiqueta_mayor} asignada a fondo (0)")
    else:
        etiqueta_mayor = None

    # Crear la máscara binaria
    mascara_binaria = np.zeros(data.shape, dtype=np.uint8)

    # Asignar 1 a todas las etiquetas excepto el fondo (0 y la etiqueta con más píxeles)
    for etiqueta, _ in etiqueta_pixel_counts.items():
        if etiqueta != etiqueta_fondo and etiqueta != etiqueta_mayor:
            mascara_binaria[data == etiqueta] = 1
            print(f"  - Aurícula: {etiqueta} -> 1")

    # Mostrar los valores únicos después del procesamiento
    unique_after, _ = np.unique(mascara_binaria, return_counts=True)
    print(f"Valores únicos después del procesamiento: {unique_after}")

    return mascara_binaria, header


def main():
    for archivo in os.listdir(ruta_segmentaciones):
        if archivo.endswith(".nrrd"):
            ruta_segmentacion = os.path.join(ruta_segmentaciones, archivo)
            print(f"\n📦 Procesando: {ruta_segmentacion}")
            try:
                # Convertir a binario manteniendo estructura
                mascara_binaria, header = convertir_a_binario(ruta_segmentacion)

                # Ruta de salida
                nombre_salida = f"binaria_{archivo}"
                ruta_salida_archivo = os.path.join(ruta_salida, nombre_salida)

                # Guardar la imagen binaria
                nrrd.write(ruta_salida_archivo, mascara_binaria, header)
                print(f"✅ Guardado: {ruta_salida_archivo}")

            except Exception as e:
                print(f"❌ Error al procesar {archivo}: {e}")


if __name__ == "__main__":
    main()
