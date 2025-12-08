import cv2
import numpy as np
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

app = FastAPI()

print("Carregando modelo de IA...")
model = YOLO('yolov8n-seg.pt') 
print("Modelo carregado!")

@app.get("/")
async def get():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Erro: Arquivo index.html não encontrado.</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # --- OTIMIZAÇÃO DE PERFORMANCE ---
            # Redimensiona para uma largura fixa (ex: 640px) para acelerar a inferência na CPU
            # Mantendo o aspect ratio
            height, width = frame.shape[:2]
            max_width = 640
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_small = cv2.resize(frame, (new_width, new_height))
            else:
                frame_small = frame

            # Inferência na imagem reduzida
            results = model(frame_small, verbose=False, retina_masks=False)
            
            pixel_count = 0
            
            # Cria máscara preta do tamanho da imagem reduzida
            mask_img = np.zeros_like(frame_small)

            if results[0].masks is not None:
                for mask in results[0].masks.xy:
                    points = np.int32([mask])
                    cv2.fillPoly(mask_img, points, (255, 255, 255))
                
                gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                pixel_count = cv2.countNonZero(gray_mask)
            
            # Codifica a resposta (Qualidade 70 é suficiente para preview)
            _, buffer = cv2.imencode('.jpg', mask_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_json({
                "image": f"data:image/jpeg;base64,{b64_img}",
                "pixels": int(pixel_count)
            })

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)