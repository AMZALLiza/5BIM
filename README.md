# Détection d'objets ( mouse et laptop) à l'aide de YOLOv5

Ce projet utilise le modèle yolov5 pour détecter des objets à partir d'une source vidéo (webcam ou un fichier vidéo).

## Prérequis

Installer les bibliothèques suivantes :

- Python
- PyTorch
- Opencv (`cv2`)
- Numpy

## Installation 

Vous pouvez installer opencv via l'une des deux commandes suivantes : 
```bash
pip install opencv-python
```
si vous avez des erreurs installer via cette commande :  

```bash
pip install opencv-python-headless
```

Vous pouvez installer PyTorch via la commande :

```bash
pip install torch torchvision torchaudio
```

## Start 

Pour lancer le projet :
```bash
python main.py
```
pour quitter le programme utiliser ```CTRL+C``` et pour quitter la capture vidéo appuyer sur la lettre  ```q```

## Exemples de résultats : 

* Détection de souris :

![Détection d'objets - Exemple 1](help/images/detect_mouse.png)

* Détection d'ordinateur : 

![Détection d'objets - Exemple 2](help/images/laptop_detect.png)