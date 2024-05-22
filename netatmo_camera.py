import aiohttp
import asyncio
import os
import cv2
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
        async with session.get(stream_url) as resp:
            if resp.status == 200:
                stream_data = await resp.text()
                print(f"Flux vidéo obtenu à partir de {stream_url}")
                print(stream_data)
                cap = cv2.VideoCapture(stream_url)
                while True:
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                       
                        cv2.imshow('Camera', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Erreur lors de la lecture du flux vidéo")
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                print(f"Erreur lors de la récupération du flux vidéo : {resp.status}")
                error_info = await resp.text()
                print(f"Message d'erreur : {error_info}")

if __name__ == "__main__":
    netatmo_client = NetatmoClient()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(netatmo_client.get_camera_data())

