import cv2
import os
import pathlib

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

clf = cv2.CascadeClassifier(str(cascade_path))


def detect_faces_and_display(image_path):
    if os.path.isdir(image_path):
        # Iterate over all image files in the folder
        for filename in os.listdir(image_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                process_image(os.path.join(image_path, filename))
    elif os.path.isfile(image_path):
        process_image(image_path)
    else:
        print("Invalid path provided.")


def process_image(image_path):
    frame = cv2.imread(image_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 255, 0), thickness=2)
        # Calculate facial area
        facial_area = width * height
        # Display facial area
        cv2.putText(frame, f"Area: {facial_area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Faces", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = "snapshots"  # Provide either an image file or a folder containing multiple images
detect_faces_and_display(image_path)
