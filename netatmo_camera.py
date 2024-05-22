import aiohttp
import asyncio
import os
import cv2
import time
from dotenv import load_dotenv

load_dotenv('config.env')

class NetatmoClient:
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        self.base_url = 'https://api.netatmo.com'

        print(f"API Key: {self.api_key}")

    async def get_camera_data(self):
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }

            async with session.get(f'{self.base_url}/api/gethomedata', headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(data)
                    for home in data.get('body', {}).get('homes', []):
                        print(f"Maison : {home.get('name')}")
                        for camera in home.get('cameras', []):
                            print(f" - Caméra : {camera.get('name')}")
                            print(f"   - URL VPN : {camera.get('vpn_url')}")
                            print(f"   - URL Local : {camera.get('local_url')}")
                            print(f"   - Etat : {camera.get('status')}")
                            print(camera)
                            stream_url = camera.get('vpn_url')
                            if stream_url:
                                stream_url_full = f"{stream_url}/live/index.m3u8"
                                await self.get_camera_stream(session, stream_url_full)
                else:
                    error_info = await resp.text()
                    print(f"Erreur lors de la récupération des données des caméras : {resp.status}")
                    print(f"Message d'erreur : {error_info}")

    async def get_camera_stream(self, session, stream_url):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        async with session.get(stream_url) as resp:
            if resp.status == 200:
                print(f"Flux vidéo obtenu à partir de {stream_url}")
                cap1 = cv2.VideoCapture(stream_url)
                cap2 = cv2.VideoCapture(0)  # Utiliser la caméra locale

                # Vérifier si les captures vidéo sont ouvertes
                if not cap1.isOpened() or not cap2.isOpened():
                    print("Erreur lors de l'ouverture d'un des flux vidéo")
                    return

                # Variables pour calculer le FPS
                start_time = time.time()
                frames = 0
                
                while True:
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()

                    if ret1 and ret2:
                        # Redimensionner les frames pour qu'elles aient la même hauteur
                        height = min(frame1.shape[0], frame2.shape[0])
                        frame1 = cv2.resize(frame1, (frame1.shape[1] * height // frame1.shape[0], height))
                        frame2 = cv2.resize(frame2, (frame2.shape[1] * height // frame2.shape[0], height))
                        
                        # Détection des visages et des yeux sur le flux de la caméra Netatmo
                        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                        faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
                        for (x, y, w, h) in faces1:
                            cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            roi_gray1 = gray1[y:y+h, x:x+w]
                            roi_color1 = frame1[y:y+h, x:x+w]
                            eyes1 = eye_cascade.detectMultiScale(roi_gray1)
                            for (ex, ey, ew, eh) in eyes1:
                                cv2.rectangle(roi_color1, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                        # Détection des visages et des yeux sur le flux de la caméra locale
                        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
                        for (x, y, w, h) in faces2:
                            cv2.rectangle(frame2, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            roi_gray2 = gray2[y:y+h, x:x+w]
                            roi_color2 = frame2[y:y+h, x:x+w]
                            eyes2 = eye_cascade.detectMultiScale(roi_gray2)
                            for (ex, ey, ew, eh) in eyes2:
                                cv2.rectangle(roi_color2, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                        # Combiner les deux frames côte à côte
                        combined_frame = cv2.hconcat([frame1, frame2])

                        # Affichage du nombre de FPS sur l'image combinée
                        frames += 1
                        end_time = time.time()
                        fps = frames / (end_time - start_time)
                        cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                        cv2.imshow('Camera', combined_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Erreur lors de la lecture d'un des flux vidéo")
                        break

                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()
            else:
                print(f"Erreur lors de la récupération du flux vidéo : {resp.status}")
                error_info = await resp.text()
                print(f"Message d'erreur : {error_info}")

if __name__ == "__main__":
    netatmo_client = NetatmoClient()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(netatmo_client.get_camera_data())
