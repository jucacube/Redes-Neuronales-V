[README (2).md](https://github.com/user-attachments/files/34557565/README.2.md)
# Reconocimiento facial con Redes Neuronales Siamesas y Triplet Loss

Práctica de la actividad **Redes Neuronales V** (Especialización en Inteligencia Artificial, Universidad de Cundinamarca).

Se implementa una **red neuronal siamesa** en **PyTorch** que verifica si dos fotografías de rostros pertenecen a la misma persona, usando **Triplet Loss**, y se evalúa en escenarios de **One-shot**, **Few-shot** y **Zero-shot Learning**.

**Autor:** Juan Camilo Cruz Beltran

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Redes_Neuronales_V_Siamesa_Facial.ipynb` | Notebook de Google Colab con todo el desarrollo, organizado por secciones |
| `README.md` | Este archivo |

> **Privacidad:** el repositorio **no incluye** las fotografías del dataset ni los comprobantes de consentimiento. Los rostros son datos biométricos y las personas autorizaron su uso solo con fines académicos.

---

## ¿Qué hace el notebook?

1. **Configuración del entorno:** instalación de `facenet-pytorch`, montaje de Google Drive y semilla fija para reproducibilidad.
2. **Dataset:** 5 personas con 10 fotografías cada una (50 imágenes), con distintas poses, expresiones y accesorios.
3. **Preprocesamiento:** detección y recorte de rostros con **MTCNN**, redimensión a 160×160 y normalización a [-1, 1]. *Data augmentation* (giro horizontal, brillo y contraste) solo en entrenamiento.
4. **Partición:** 4 personas con 7 fotos de entrenamiento y 3 de prueba; 1 persona reservada que no participa en el ajuste fino.
5. **Tripletas:** generación de (ancla, positivo, negativo), con negativos aleatorios y con **negativos difíciles** (*hard negative mining*).
6. **Arquitectura siamesa:** **InceptionResnetV1** preentrenada en VGGFace2, con pesos compartidos en las tres ramas y ajuste fino de las últimas capas (`block8`, `last_linear`, `last_bn`). Cada rostro se convierte en un *embedding* de 512 dimensiones.
7. **Triplet Loss:** `nn.TripletMarginLoss` con margen 0.5, optimizador Adam (lr = 1e-4), lotes de 16 y 10 épocas.
8. **Evaluación:** verificación de pares con umbral de distancia, One-shot (1 referencia por persona) y Few-shot (3 referencias por persona, prototipo promedio).

---

## Resultados

| Escenario | Resultado |
|---|---|
| Distancia media, misma persona | 0.418 |
| Distancia media, personas distintas | 1.561 |
| Verificación de pares (66 pares, umbral 0.623) | 100 % |
| One-shot (21 consultas) | 21 / 21 |
| Few-shot (19 consultas) | 19 / 19 |
| Persona no vista en el ajuste fino (One-shot) | 9 / 9 |

### Cómo interpretar estos resultados

- La **Triplet Loss fue casi cero desde la primera época**. El modelo base, preentrenado en VGGFace2, ya separa muy bien las identidades, por lo que el buen desempeño se debe sobre todo al preentrenamiento y no al ajuste fino.
- El **dataset es pequeño** (5 personas), así que el 100 % no garantiza el mismo desempeño con más personas, otras cámaras o iluminación difícil.
- El umbral se eligió con los mismos pares de prueba, lo que puede inflar ligeramente la *accuracy*.

---

## Cómo ejecutarlo

1. Abre el notebook en **Google Colab** (Archivo → Abrir cuaderno → GitHub, o sube el `.ipynb`).
2. Activa la **GPU**: Entorno de ejecución → Cambiar tipo de entorno → GPU.
3. Prepara tu propio dataset en Google Drive, con una carpeta por persona:

   ```
   MyDrive/
     dataset/
       persona_01/   (fotos .jpg o .png)
       persona_02/
       ...
       persona_05/
   ```

   Usa únicamente fotografías propias o de personas que hayan autorizado su uso.
4. Ajusta las rutas `RUTA` y `SALIDA` en la celda de configuración si tus carpetas se llaman distinto (en Drive se distinguen mayúsculas).
5. Ejecuta todas las celdas en orden (Entorno de ejecución → Ejecutar todo).

### Dependencias

Colab ya trae `torch`, `torchvision`, `numpy`, `Pillow` y `matplotlib`. Solo se instala:

```bash
pip install facenet-pytorch --no-deps
```

Se usa `--no-deps` para no sobrescribir la versión de PyTorch de Colab.

### Nota de compatibilidad

`facenet-pytorch` llama a `torchvision._is_tracing`, que las versiones recientes de torchvision ya no exponen y provoca un `AttributeError` dentro de `batched_nms`. El notebook lo resuelve con este parche, antes de crear el detector:

```python
torchvision._is_tracing = lambda: False
```

---

## Tecnologías

Python · PyTorch · torchvision · facenet-pytorch (MTCNN e InceptionResnetV1) · NumPy · Matplotlib · Pillow · Google Colab

## Referencias principales

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*, 815–823.
- Cao, Q., Shen, L., Xie, W., Parkhi, O. M., & Zisserman, A. (2018). VGGFace2: A dataset for recognising faces across pose and age. *IEEE FG 2018*, 67–74.
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE Signal Processing Letters, 23*(10), 1499–1503.
- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*.
- Esler, T. (s. f.). *facenet-pytorch* [Software]. https://github.com/timesler/facenet-pytorch
