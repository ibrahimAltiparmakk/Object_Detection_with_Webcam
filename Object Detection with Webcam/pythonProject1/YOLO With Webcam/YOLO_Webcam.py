import math

from PIL.ImageOps import scale
from ultralytics import YOLO  # gerçek zamanlı nesne tespiti
import  cv2 #  bilgisayar görüşü uygulamaları için tasarlanmış açık kaynaklı bir kütüphane
import cvzone # bilgisayar görüşü görevlerini daha az kod ile daha hızlı bir şekilde gerçekleştirmek için basitleştirilmiş araçlar sağlamaktır.
import matplotlib


cap = cv2.VideoCapture(0) # bilgisayarın varsayılan kamerasından (genellikle 0 ilk entegre webcam anlamına gelir) video yakalamak için
cap.set(3,1280) # kamera çerçevesinin genişliğini 640 piksel olarak ayarlar. 3, genişlik ayarının kodunu temsil eder.
cap.set(4,720) # kamera çerçevesinin yüksekliğini 480 piksel olarak ayarlar. 4, yükseklik ayarının kodunu temsil eder.

model = YOLO("../YOLO_Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success , img = cap.read() #  kameradan bir görüntü çerçevesi okur. read() fonksiyonu iki değer döndürür: success (bir Boolean, çerçevenin başarıyla okunup okunmadığını belirtir) ve img (okunan görüntü çerçevesi).
    results = model(img,stream=True) # model(img, stream=True) ifadesi, YOLO modelini kullanarak, img üzerinde nesne tespiti yapar. stream=True parametresi, modelin video akışı modunda çalıştığını belirtir, yani sürekli gelen görüntüler üzerinde analiz yapılması gerektiğini ifade eder.
    for r in results: # Model tarafından döndürülen sonuçlar üzerinde iterasyon yapılır. Her bir r nesnesi, tespit edilen nesnelerle ilgili bilgiler içerir.
        boxes = r.boxes # tespit edilen nesnelerin her biri için sınırlayıcı kutuların (bounding boxes) bir listesini içerir.
        for box in boxes: # Her bir kutu için döngü başlatılır. Bu kutular, tespit edilen nesnelerin görüntü üzerindeki konumlarını temsil eder.
            x1 ,y1 ,x2 ,y2 = box.xyxy[0] # box.xyxy ifadesi, kutunun köşe koordinatlarını içerir. x1, y1 kutunun sol üst köşesi, x2, y2 ise sağ alt köşesi anlamına gelir.
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2) # Koordinatlar genellikle ondalık olarak dönebilir; bu yüzden koordinatları tam sayıya çevirmek gereklidir.
            print(x1 ,y1 ,x2 ,y2) # Kutunun koordinatları konsola yazdırılır.
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3) # cv2.rectangle() fonksiyonu, img üzerine bir dikdörtgen çizer. (x1, y1) ve (x2, y2) dikdörtgenin köşe koordinatlarıdır.(255, 0, 255) renk kodu, 3 ise dikdörtgenin çizgi kalınlığıdır.

            # confidence (güven endexi)
            conf = math.ceil((box.conf[0]*100))/100
            # math.ceil(): Elde edilen değeri yukarı yuvarlar. Bu işlem, özellikle düşük güven skorlarındaki küçük değişimlerin daha belirgin olmasını sağlar.
            # box.conf[0]: Modelin, tespit edilen nesneye atadığı güven skorunu alır. Bu skor 0 ile 1 arasında bir değerdir.
            # (box.conf[0]*100): Güven skorunu yüzdelik bir değere dönüştürmek için 100 ile çarpılır.
            # /100: Yuvarlanmış değeri tekrar orijinal aralığa (0.00 ile 1.00 arası) çekmek için 100'e bölünür

            print(conf)


            # Class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1,thickness=1)
            # box.cls[0]: Tespit edilen nesnenin sınıf indeksini içerir. Örneğin, 0 insan, 1 bisiklet anlamına gelebilir.
            # int(box.cls[0]): Sınıf indeksini tam sayıya dönüştürür.
            # cvzone.putTextRect(): İlgili sınıf adını ve güvenilirlik skorunu, belirlenen koordinatlara, belirlenen ölçek ve kalınlıkta görüntü üzerine yazdırır. max(0, x1) ve max(35, y1) kullanımı, yazının görüntünün dışına çıkmamasını sağlar; özellikle sınırlar çok düşük olduğunda kullanışlıdır.
            # max(0, x1): Bu ifade, x1 değerinin 0'dan küçük olmamasını garanti eder. Yani, x1 pozitif bir sayı veya sıfır ise x1'i, negatif ise 0'ı döndürür.
            # max(35, y1): Benzer şekilde, bu ifade y1 değerinin 35'ten küçük olmamasını garanti eder.

    cv2.putText(img, "Kamerayi kapatmak icin 'q' tusuna basin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Image",img) # okunan görüntüyü bir pencerede gösterir. "Image" pencerenin başlığıdır ve img gösterilen görüntüdür.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
# cap nesnesi, cv2.VideoCapture() ile oluşturulan bir video yakalama nesnesidir. Bu nesne, bilgisayarın kamerası gibi bir video kaynağından veri akışını yönetir.
# cap.release() fonksiyonu, bu video yakalama nesnesini serbest bırakır. Yani, kamera veya video dosyası gibi kaynakları işletim sistemine geri verir.
cv2.destroyAllWindows()
# cv2.imshow() ile açılan her pencere, işletim sisteminde belirli kaynaklar kullanır. Bu pencereler, görüntüleme amaçlı kullanılır.
#cv2.destroyAllWindows() fonksiyonu, OpenCV tarafından oluşturulan tüm pencereleri kapatır ve bu pencereler tarafından kullanılan kaynakları serbest bırakır.