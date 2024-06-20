import cv2  # Importer OpenCV pour le traitement d'image
import numpy as np  # Importer numpy pour les calculs numériques
import torch  # Importer PyTorch pour le chargement du modèle yolov5

class ObjectDetector:
    def __init__(self, model_path, names_path):
        """
        Initialise le détecteur d'objets avec le modèle yolov5 et les noms de classes.
        les arguments :
        - model_path: Chemin vers le fichier de poids du modèle YOLOv5
        - names_path: Chemin vers le fichier contenant les noms des classes
        """
        # Charge le modèle yolov5 depuis le chemin spécifié
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        
        # Charge les noms de classes à partir du fichier spécifié
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Définit la liste des classes cibles à détecter
        self.target_classes = ['laptop', 'mouse']  # ici on met les objets à détecter, on peut rajouter plus

    def load_video(self, source=0):
        """
        Initialise la capture vidéo à partir de la source spécifiée.
        """
        self.cap = cv2.VideoCapture(source)  # Initialise l'objet de capture vidéo

    def process_frame(self, frame):
        """
        Effectue l'inférence d'objets sur le cadre donné et dessine les boîtes englobantes et les labels.
        """
        # Effectue l'inférence avec le modèle yolov5 sur le cadre actuel
        results = self.model(frame)
        predictions = results.pandas().xyxy[0]  # Récupère les prédictions d'objets détectés.

        # Parcourt les prédictions et dessine les boîtes englobantes et les labels correspondants.
        for _, pred in predictions.iterrows():
            label = pred['name']
            confidence = pred['confidence']
            if label in self.target_classes and confidence > 0.5:
                x1, y1, x2, y2 = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessine un rectangle autour de l'objet détecté
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return frame

    def run(self):
        """
        Lance le traitement de la capture vidéo en temps réel et affiche les résultats de la détection.
        """
        try:
            while True:
                ret, frame = self.cap.read()  # Lit un cadre de la vidéo capturée.
                if not ret:
                    print("Erreur de capture vidéo")
                    break
                
                processed_frame = self.process_frame(frame)  # Traite le cadre pour la détection d'objets
                cv2.imshow('Detection', processed_frame)  # Affiche le cadre traité avec les résultats de la détection
                
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Attend l'appui sur la touche 'q' pour quitter
                    break
        
        except KeyboardInterrupt:
            print("Interruption détectée. Fin de la détection")
        
        finally:
            # Libère la capture vidéo et détruit toutes les fenêtres OpenCV.
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Chemin vers les fichiers de poids et de noms de classes de yolov5.
    model_path = './yolov5/yolov5s.pt'  
    names_path = './help/coco.names' 

    # Initialise le détecteur d'objets et démarre la capture vidéo.
    detector = ObjectDetector(model_path, names_path)
    detector.load_video(0)  # 0 pour la webcam ( 1 ou 2 si le 0 ne fonctionne pas)
    detector.run()
