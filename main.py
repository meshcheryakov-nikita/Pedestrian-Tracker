import gradio as gr
import cv2
import numpy as np
from yolov5 import YOLOv5

class TrafficLightDetector:
    def __init__(self, traffic_light_region):
        """
        Инициализирует детектор светофора.
        """
        self.traffic_light_region = traffic_light_region

    def detect_traffic_light_state(self, frame):
        """
        Определяет состояние светофора на изображении.
        """
        x, y, w, h = self.traffic_light_region
        traffic_light_roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

        lower_red = (0, 70, 50)
        upper_red = (10, 255, 255)
        lower_green = (40, 40, 40)
        upper_green = (70, 255, 255)

        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_pixel_count = cv2.countNonZero(red_mask)
        green_pixel_count = cv2.countNonZero(green_mask)

        if red_pixel_count > green_pixel_count and red_pixel_count > 50:
            return "red"
        elif green_pixel_count > red_pixel_count and green_pixel_count > 50:
            return "green"
        else:
            return "unknown"


class PedestrianDetector:
    def __init__(self, model_path, confidence_threshold=0.5, nms_threshold=0.4, device="cpu"):
        """
        Инициализирует детектор пешеходов с использованием YOLOv5.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Загрузка модели YOLOv5
        self.model = YOLOv5(model_path, device=device)

    def detect_pedestrians(self, frame):
        """
        Детектирует пешеходов на кадре с использованием YOLOv5.
        """
        results = self.model.predict(frame)

        boxes = []
        for box in results.pred[0]:  # Доступ к обнаружениям через results.pred[0]
            x1, y1, x2, y2, confidence, class_id = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w = x2 - x1
            h = y2 - y1
            if confidence > self.confidence_threshold and class_id == 0:  # 0 - индекс класса "person"
                boxes.append([x1, y1, w, h])

        return boxes


class PedestrianTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0

    def add_tracker(self, frame, bbox):
        """
        Добавляет новый трекер для пешехода.
        """
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        self.trackers[self.next_id] = tracker
        self.next_id += 1

    def update_trackers(self, frame):
        """
        Обновляет все трекеры и возвращает результаты.
        """
        results = {}
        for id, tracker in list(self.trackers.items()):
            success, bbox = tracker.update(frame)
            if success:
                results[id] = bbox
            else:
                del self.trackers[id]  # Удаляем потерянный трекер
        return results


def iou(box1, box2):
    """
    Вычисляет Intersection over Union (IoU) для двух ограничивающих рамок.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x = max(x1, x2)
    intersection_y = max(y1, y2)
    intersection_w = min(x1 + w1, x2 + w2) - intersection_x
    intersection_h = min(y1 + h1, y2 + h2) - intersection_y

    if intersection_w < 0 or intersection_h < 0:
        return 0

    intersection_area = intersection_w * intersection_h
    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area
    return iou

class ViolationDetector:
    def __init__(self, crosswalk_area, min_iou_threshold=0.5):
        """
        Инициализирует модуль анализа нарушений.
        """
        self.crosswalk_area = crosswalk_area
        self.min_iou_threshold = min_iou_threshold
        self.pedestrian_entered_crosswalk = {}  # Словарь для отслеживания входа пешеходов в зону перехода

    def detect_violation(self, traffic_light_state, tracker_results):
        """
        Детектирует нарушение - переход пешехода на красный свет.
        """
        violators = []

        for id, bbox in tracker_results.items():
            if iou(bbox, self.crosswalk_area) > self.min_iou_threshold:
                # Если пешеход вошел в зону перехода
                if id not in self.pedestrian_entered_crosswalk:
                    self.pedestrian_entered_crosswalk[id] = traffic_light_state  # Записываем состояние светофора при входе

                # Если светофор был красным при входе
                if self.pedestrian_entered_crosswalk[id] == "red":
                    violators.append(id)
            else:
                # Если пешеход вышел из зоны перехода, удаляем его из словаря
                if id in self.pedestrian_entered_crosswalk:
                    del self.pedestrian_entered_crosswalk[id]

        return violators


# Функция для обработки одного кадра
def process_frame(frame, traffic_light_region_str, crosswalk_area_str):
    global tracker

    # Преобразование строковых координат в кортежи
    traffic_light_region = tuple(map(int, traffic_light_region_str.split(',')))
    crosswalk_area = tuple(map(int, crosswalk_area_str.split(',')))

    traffic_light_state = TrafficLightDetector(traffic_light_region).detect_traffic_light_state(frame)
    pedestrian_bboxes = PedestrianDetector("yolov5s.pt").detect_pedestrians(frame)
    tracker_results = tracker.update_trackers(frame)

    # Добавление новых трекеров
    for bbox in pedestrian_bboxes:
        is_tracked = False
        for tracked_bbox in tracker_results.values():
            if iou(bbox, tracked_bbox) > 0.5:
                is_tracked = True
                break
        if not is_tracked:
            tracker.add_tracker(frame, bbox)

    # Анализ нарушений
    violators = ViolationDetector(crosswalk_area).detect_violation(traffic_light_state, tracker_results)

    # --- Код для визуализации ---
    # Отображение области светофора черной рамкой
    x, y, w, h = traffic_light_region
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Отображение области пешеходного перехода белой рамкой
    x, y, w, h = crosswalk_area
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    for id, bbox in tracker_results.items():
        x, y, w, h = [int(v) for v in bbox]
        color = (0, 0, 255) if id in violators else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Вывод состояния светофора
    cv2.putText(frame, f"Светофор: {traffic_light_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

# Создание интерфейса Gradio
iface = gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Image(source="webcam", streaming=True, label="Видеопоток"),
        gr.Textbox(label="Область светофора (x,y,w,h)", placeholder="100,50,30,60"),
        gr.Textbox(label="Область пешеходного перехода (x,y,w,h)", placeholder="200,300,100,50")
    ],
    outputs=gr.Image(label="Результат"),
    title="Детектирование пешеходов",
    description="Загрузите видео или используйте веб-камеру для детектирования пешеходов, нарушающих ПДД."
)

# Инициализация трекера
tracker = PedestrianTracker()

# Запуск интерфейса
iface.launch()

