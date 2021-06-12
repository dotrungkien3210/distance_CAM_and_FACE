'''
ý tưởng của bài này là dựa trên một reference image tức là ảnh mẫu
cùng với khoảng cách của ảnh mẫu(reference image) tới khuôn mặt
sau đó ta tiến hành đọc video từ camera và đo lại phần diện tích
chiếm chỗ của khuôn mặt trên camera, nếu phần diện tích càng lớn
thì mặt càng gần cam và nếu phần diện tích càng nhỏ thì tức là
mặt đang càng xa cam.
https://www.tutorialspoint.com/opencv/opencv_face_detection_in_picture.htm

'''
import cv2
# khoảng cách từ khuôn mặt đến camera đo bằng thước
Known_distance = 30  # Inches
# chiều dài khuôn mặt của em
Known_width = 5.7  # Inches

# Chọn màu để vẽ đường bao cho face bên dưới
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX

# Camera Object
cap = cv2.VideoCapture(0)  # tức là đang dùng camera của máy tính
Distance_level = 0

# Define the codec and create VideoWriter object
# lưu lại video vào trong cùng folder với tên output21.mp4
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output21.mp4', fourcc, 30.0, (640, 480))

# nhận dạng khuôn mặt
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

''' hàm này tính toán Focal Length(độ dài tiêu cự),
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE 
    :param1 Measure_Distance(int): khoảng cách đo từ vật thể tới camera while Capturing Reference image

    :param2 Real_Width(int): chiều rộng thực tế trong thế giới thực
    :param3 Width_In_Image(int):
     độ dài của vật thể trên khung hình chia cho
     --------------------------------------------------------------
     image in our case in the reference image(found by Face detector)
    :retrun Focal_Length(Float):
'''
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


''' hàm này ước tính khoảng cách giữa đối tượng và máy ảnh sử dụng các tham số
    (Focal_Length, Actual_object_width, Object_width_in_the_image)
    :param1 Focal_length(float): return by the Focal_Length_Finder function

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    :return Distance(float) : distance Estimated  
'''
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

'''
    hàm này nhận dạng khuôn mặt, vẽ hình chữ nhật và tính khoảng cách
    :param1 Image(Mat): simply the frame 
    :param2 Call_Out(bool): bạn có muốn hiển thị khoảnh cách và vẽ vuông gương mặt trước màn hình không
    :param3 Distance_Level(int): thay đổi các cạnh bo vào khuôn mặt khi mà khoảng cách thay đổi
    :return1  face_width(int): it is width of face in the frame which allow us to calculate the distance and find focal length
    :return2 face(list): length of face and (face paramters)
    :return3 face_center_x: face centroid_x coordinate(x)
    :return4 face_center_y: face centroid_y coordinate(y)
'''
def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    # chuyển ảnh đầu vào sang màu xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    '''
    đoạn code này vẽ các đường bo tròn cho hình chữ nhật
    LLV chỉ là đoạn bo cong xuống ở trái phải 2 bên
    '''
    for (x, y, h, w) in faces:
        line_thickness = 2
        LLV = int(h * 0.12)
        cv2.line(image, (x, y+LLV), (x+w, y+LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y+h), (x+w, y+h), (GREEN), line_thickness)
        cv2.line(image, (x, y+LLV), (x, y+LLV+LLV), (GREEN), line_thickness)
        cv2.line(image, (x+w, y+LLV), (x+w, y+LLV+LLV),(GREEN), line_thickness)
        cv2.line(image, (x, y+h), (x, y+h-LLV), (GREEN), line_thickness)
        cv2.line(image, (x+w, y+h), (x+w, y+h-LLV), (GREEN), line_thickness)

        face_width = w
        face_center = []
        # Drwaing circle at the center of the face
        face_center_x = int(w/2)+x
        face_center_y = int(h/2)+y
        if Distance_level < 10:
            Distance_level = 10


        # ở video thì set True cho phép hiển thị thêm distance
        if CallOut == True:
            # cv2.line(image, (x,y), (face_center_x,face_center_y ), (155,155,155),1)
            cv2.line(image, (x, y-11), (x+180, y-11), (ORANGE), 28)
            cv2.line(image, (x, y-11), (x+180, y-11), (YELLOW), 20)
            cv2.line(image, (x, y-11), (x+Distance_level, y-11), (GREEN), 18)

            # cv2.circle(image, (face_center_x, face_center_y),2, (255,0,255), 1 )
            # cv2.circle(image, (x, y),2, (255,0,255), 1 )

        # face_x = x
        # face_y = y

    return face_width, faces, face_center_x, face_center_y




# Đọc ảnh mà ta lấy làm ảnh đại diện để đo
ref_image = cv2.imread("Ref_image.png")
# truyền ảnh ref vào và bắt đầu tính toán khoảng cách
ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(
    Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)
cv2.imshow("ref_image", ref_image)



while True:
    _, frame = cap.read()
    # calling face_data function
    # Distance_leve =0

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(
        frame, True, Distance_level)


    # tìm khoảng cách bằng cách gọi hàm Distance finder
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:

            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            # Drwaing Text on the screen
            Distance_level = int(Distance)

            cv2.putText(frame, f"Distance {Distance} cm",
                        (face_x-6, face_y-6), fonts, 0.5, (BLACK), 2)
    cv2.imshow("frame", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
