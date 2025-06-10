import flet as ft
import cv2
import base64
import threading
import time
import numpy as np
import mediapipe as mp


def overlay_with_alpha(background_img, overlay_img, x, y):
    h_overlay, w_overlay, _ = overlay_img.shape
    h_bg, w_bg, _ = background_img.shape
    x, y = int(x), int(y)
    y1, y2 = max(0, y), min(y + h_overlay, h_bg)
    x1, x2 = max(0, x), min(x + w_overlay, w_bg)
    roi_bg = background_img[y1:y2, x1:x2]
    roi_h, roi_w = roi_bg.shape[:2]
    if roi_h > 0 and roi_w > 0:
        overlay_cropped = overlay_img[:roi_h, :roi_w]
        alpha = overlay_cropped[:, :, 3] / 255.0
        overlay_rgb = overlay_cropped[:, :, :3]
        for c in range(0, 3):
            roi_bg[:, :, c] = (overlay_rgb[:, :, c] * alpha) + (roi_bg[:, :, c] * (1.0 - alpha))
    return background_img

class CameraApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.is_running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.page.add(ft.Text("Erro: Não foi possível encontrar a webcam.", color=ft.Colors.RED, size=20))
            self.is_running = False
            return

        try:
            self.overlay_png = cv2.imread('assets/computer.png', cv2.IMREAD_UNCHANGED)
            if self.overlay_png is None or self.overlay_png.shape[2] != 4:
                raise ValueError("A imagem 'assets/computer.png' não foi encontrada ou não possui transparência (canal alfa).")
        except Exception as e:
            self.page.add(ft.Text(f"Erro: {e}", color=ft.Colors.RED, size=16))
            self.is_running = False
            return

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        placeholder_pixel = np.zeros((1, 1, 4), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', placeholder_pixel)
        b64_string_placeholder = base64.b64encode(buffer).decode('utf-8')
        
        self.camera_image = ft.Image(src_base64=b64_string_placeholder, fit=ft.ImageFit.CONTAIN, expand=True)
        self.stack = ft.Stack(controls=[self.camera_image], expand=True)

    def update_camera_thread(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                frame_h, frame_w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                        center_x = int(((wrist.x + middle_finger_mcp.x) / 2) * frame_w)
                        center_y = int(((wrist.y + middle_finger_mcp.y) / 2) * frame_h)

                        overlay_w = 150
                        overlay_h = int(overlay_w * (self.overlay_png.shape[0] / self.overlay_png.shape[1]))
                        resized_overlay = cv2.resize(self.overlay_png, (overlay_w, overlay_h))

                        draw_x = center_x - (overlay_w // 2)
                        draw_y = center_y - (overlay_h // 2)

                        frame = overlay_with_alpha(frame, resized_overlay, draw_x, draw_y)

                _, buffer = cv2.imencode('.jpg', frame)
                b64_string = base64.b64encode(buffer).decode('utf-8')
                self.camera_image.src_base64 = b64_string
                self.page.update()
                time.sleep(1/60)

            except Exception as e:
                print(f"Erro no loop da câmera: {e}")
        
        self.hands.close()
        self.cap.release()
        print("Recursos liberados.")

    def cleanup(self):
        self.is_running = False

def main(page: ft.Page):
    page.title = "AR com Detecção de Mão no Windows"
    page.window_width = 800
    page.window_height = 600
    page.padding = 0
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    app = CameraApp(page)

    def on_window_event(e):
        if e.data == "close":
            app.cleanup()
            page.window_destroy()

    page.on_window_event = on_window_event

    if app.is_running:
        page.add(app.stack)
        camera_thread = threading.Thread(target=app.update_camera_thread, daemon=True)
        camera_thread.start()

ft.app(target=main, assets_dir="assets")