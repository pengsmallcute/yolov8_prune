from ultralytics import YOLO

model = YOLO(r'C:\Users\fan\Desktop\prune\ultralytics\ultralytics\cfg\models\v8\yolov8.yaml')


if __name__=='__main__':
    
   results = model.train(
                        data=r'C:\Users\fan\Desktop\prune\ultralytics\dataset\data.yaml',
                        epochs=50, 
                        imgsz=640, 
                        batch=32, 
                        device='0',
                        project = r"C:\Users\fan\Desktop\prune\ultralytics\runs\prune",
                        workers = 0,
                        aim = 0.3,
                        prune = True,
                        amp = False,
                        iterative_steps = 5,
                        name = 'prune'
                     )