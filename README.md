# distance_yolo (CPU, Ultralytics YOLOv8)

YOLO로 검출된 객체에 대해 카메라 FOV 기반으로 거리를 추정합니다. 객체의 실세계 대표 크기를 `class_sizes.json`에서 읽어 사용합니다.

## 구성
- `yolo_distance.py`: 추론 + 거리 계산
- `config_yolo.json`: 설정 파일
- `class_sizes.json`: 클래스별 대표 크기(mm)
- `requirements.txt`: 의존성

## 설치
```
cd distance_yolo
python -m pip install -r requirements.txt
```

## 실행
```
python yolo_distance.py
```

## 설정 (config_yolo.json)
- `engine`: "cpu"
- `model_path`: 가중치 경로(예: yolov8x.pt)
- `source`: "camera" 또는 비디오 경로
- `camera_number`: 카메라 인덱스
- `input_size`: [640, 640] (필요 시 전처리 참고용)
- `fov_degree` × `fov_correction_factor`: 수평 FOV(도) 계산
- `use_width`: true=가로폭으로 거리 계산, false=세로높이
- `class_sizes_file`: 클래스 크기 매핑 파일명
- `label_overrides_mm`: 우선 적용되는 클래스별 크기(mm)

거리 정확도는 FOV 및 클래스 크기 설정에 영향을 받습니다. 환경에 맞게 조정하세요.