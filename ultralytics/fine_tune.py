from ultralytics.models.yolo.detect import DetectionTrainer



if __name__=='__main__':

    args = dict(
                model=r'C:\Users\fan\Desktop\prune\ultralytics\runs\prune\prune\weights\last.pt',
                data=r'C:\Users\fan\Desktop\prune\ultralytics\dataset\data.yaml',
                epochs=100, 
                imgsz=640, 
                batch=32, 
                device='0',
                project = r"C:\Users\fan\Desktop\prune\ultralytics\runs\fine_tune",
                fine_tune = True,
                name = "exp"
        )
    
    model = DetectionTrainer(overrides=args)


    model.train()

    