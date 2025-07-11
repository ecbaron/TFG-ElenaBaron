import os
import glob
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from skimage import measure
import trimesh
from vedo import Mesh as VedoMesh, Plotter
import time

# Carpeta de entrada
input_dir = r"C:\Users\elena\Documents\TFG\segmentaciones\reescaladas_definitivo\mask_cut_def\MASCARA_BINARIA_SIN_ORDENAR"
output_base_dir = r"C:\Users\elena\Documents\TFG\mallas 3D\cascara_nueva"

# Parámetros
iterations = 4

# Buscar todos los ficheros .nrrd de máscaras
nrrd_paths = glob.glob(os.path.join(input_dir, "*.nrrd"))

for nrrd_path in nrrd_paths:
    paciente_id = os.path.basename(nrrd_path).split("_SegPaciente")[-1].split("_")[0]
    print(f"\n🩺 Procesando paciente {paciente_id}...")

    # Crear carpeta de salida
    out_dir = os.path.join(output_base_dir, f"paciente{paciente_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Definir rutas de salida
    path_cavidad = os.path.join(out_dir, f"paciente{paciente_id}_cavidad.stl")
    path_miocardio = os.path.join(out_dir, f"pared_miocardio_{paciente_id}.stl")
    output_image_path = os.path.join(out_dir, f"superposicion_{paciente_id}.png")
    output_txt = os.path.join(out_dir, f"resumen_paciente_{paciente_id}.txt")

    # Cargar imagen y datos espaciales
    img = sitk.ReadImage(nrrd_path)
    mask = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()[::-1]  # (z, y, x)
    origin = np.array(img.GetOrigin())

    # Binarizar
    mask_bin = (mask > 0).astype(np.uint8)

    # GENERAR STL DE CAVIDAD SI NO EXISTE
    if not os.path.exists(path_cavidad):
        verts_cav, faces_cav, _, _ = measure.marching_cubes(mask_bin, level=0.5, spacing=spacing)
        verts_cav_real = verts_cav + origin
        malla_cavidad_trimesh = trimesh.Trimesh(vertices=verts_cav_real, faces=faces_cav, process=True)
        if malla_cavidad_trimesh.volume < 0:
            malla_cavidad_trimesh.invert()
        malla_cavidad_trimesh.export(path_cavidad)
        print(f"Cavidad STL generada: {path_cavidad}")
    else:
        print("Cavidad STL ya existe.")

    # DILATACIÓN → MIOMCARDIO
    dilatada = binary_dilation(mask_bin, iterations=iterations)
    pared_miocardio = (dilatada & (~mask_bin)).astype(np.uint8)

    # Marching Cubes pared miocárdica
    verts, faces, _, _ = measure.marching_cubes(pared_miocardio, level=0.5, spacing=spacing)
    verts_real = verts + origin
    mesh_miocardio = trimesh.Trimesh(vertices=verts_real, faces=faces, process=True)
    if mesh_miocardio.volume < 0:
        mesh_miocardio.invert()

    # Guardar malla miocardio
    mesh_miocardio.export(path_miocardio)
    print(f"Malla miocardio STL exportada: {path_miocardio}")

    # Calcular métricas
    mean_spacing_mm = np.mean(spacing)
    grosor_mm = iterations * mean_spacing_mm
    volumen_miocardio = mesh_miocardio.volume
    area_miocardio = mesh_miocardio.area

    malla_cavidad_trimesh = trimesh.load(path_cavidad, process=True)
    if malla_cavidad_trimesh.volume < 0:
        malla_cavidad_trimesh.invert()
    volumen_cavidad = malla_cavidad_trimesh.volume
    area_cavidad = malla_cavidad_trimesh.area

    # Guardar resumen
    with open(output_txt, "w") as f:
        f.write(f"Resumen de paciente {paciente_id}\n")
        f.write(f"Grosor simulado del miocardio: {grosor_mm:.2f} mm\n")
        f.write(f"Volumen miocardio (mm³): {volumen_miocardio:.2f}\n")
        f.write(f"Superficie miocardio (mm²): {area_miocardio:.2f}\n")
        f.write(f"Volumen cavidad (mm³): {volumen_cavidad:.2f}\n")
        f.write(f"Superficie cavidad (mm²): {area_cavidad:.2f}\n")
        f.write(f"Ratio volumen miocardio/cavidad: {volumen_miocardio / volumen_cavidad:.2f}\n")

    # VISUALIZACIÓN
    try:
        malla_cavidad = VedoMesh(path_cavidad).color("cyan").opacity(1.0)
        malla_miocardio = VedoMesh(path_miocardio).color("magenta").opacity(0.2)
        # Cargar mallas para visualización
        # Crear visualizador
        plotter = Plotter(title=f"Paciente {paciente_id}", axes=1, bg="white", size=(1000, 800))

        # Ajustar la cámara para que coincida con la vista anatómica como en Slicer
        plotter.camera.SetPosition(-800, -100, 300)    # Desde lateral izquierda + posterior + arriba
        plotter.camera.SetFocalPoint(0, 0, 0)
        plotter.camera.SetViewUp(0, 0, 1)              # Z hacia arriba

        # Mostrar, guardar y cerrar
        plotter.show(malla_cavidad, malla_miocardio, f"Superposición {paciente_id}", interactive=False)
        plotter.screenshot(output_image_path)
        plotter.close()

        print(f"Imagen guardada en: {output_image_path}")
    except Exception as e:
        print(f"No se pudo generar la imagen: {e}")

    time.sleep(0.5)  # Espera opcional

print("\n🎉 Procesamiento de todos los pacientes completado.")
