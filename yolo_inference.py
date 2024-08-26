from ultralytics import YOLO


model = YOLO('models/best_new.pt')


results = model.predict('input_videos/23.mp4', save=True)


print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)
