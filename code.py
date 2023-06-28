import face_recognition
import cv2
import numpy as np
import boto3
import datetime

s3 = boto3.client('s3')

bucket_name ='facerecognitonmodelimages1'
file_name1 = 'Sai kalyan.jpg'
file_name2 = 'Rana.jpg'
file_name3 = 'Ravi_sir.jpg'



s3.download_file(bucket_name, file_name1, '/tmp/Sai kalyan.jpg')
s3.download_file(bucket_name, file_name2, '/tmp/Rana.jpg')
s3.download_file(bucket_name, file_name3, '/tmp/Ravi_sir.jpg')


video_capture = cv2.VideoCapture(0)


Sai_image = face_recognition.load_image_file('/tmp/Sai kalyan.jpg')
Sai_face_encoding = face_recognition.face_encodings(Sai_image)[0]

Rana_image = face_recognition.load_image_file('/tmp/Rana.jpg')
Rana_face_encoding = face_recognition.face_encodings(Rana_image)[0]

Ravi_boda_sir_image = face_recognition.load_image_file('/tmp/Ravi_sir.jpg')
Ravi_boda_sir_face_encoding = face_recognition.face_encodings(Ravi_boda_sir_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    Sai_face_encoding,
    Rana_face_encoding,
    Ravi_boda_sir_face_encoding,
]
known_face_names = [
    'Sai Kalyan',
    'Ranadeep',
    'Ravi',
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    # ....
    # your code here
    # ....

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Save the recognized face name and time to a file
        if name != "Unknown":
            now = datetime.datetime.now()
            with open("recognized_faces.txt", "a") as f:
                f.write("{},{}\n".format(now, name, frame))

    # Display the resulting image
    cv2.imshow('Video', frame)

    # ....
    # your code here
    # ....
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()