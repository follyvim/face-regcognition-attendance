import string
import re
from tkinter import messagebox
import tkinter.ttk as ttk
from tkinter import *
import tkinter as tk
import pandas as pd
import csv
import cv2
import os
from os import path
import pickle
import face_recognition
from imutils import paths
import numpy as np
import datetime
import time
from datetime import time, datetime, date
import warnings
warnings.filterwarnings('ignore')

dd = pd.Timestamp(datetime.today())
dd = dd.day_name()
today = date.today()
d3 = today.strftime("%m/%d/%y")
name = input('your full name').upper()
course = input('course').upper()
cohort = input('your cohort').upper()
path1 = f'/Users/folas/Downloads/personal project/school_att/school_pic/{name}'
filename = 'Attendence3.csv'
try:
    os.mkdir(path1)
except OSError:
    print("Creation of the directory %s failed" % path1)
else:
    print("Successfully created the directory %s " % path1)
print('###############')
print('wait for capturing')
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
while 1:
    cont = 1
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)
    for x, y, w, h in faces:
        frame = cv2.rectangle(gray, (x, y), (x+w, y+h), (128, 128, 128), 3)
        pathm = path.join(path1, f'{name}.jpg')
        cv2.imwrite(pathm, frame)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(10)
    if key == ord('d'):
        break
video.release()
cv2.destroyAllWindows()
if os.path.exists(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates(keep='first')
    data = {
        "Full Name": name,
        "Course": course,
        "Cohort": cohort,
        'Day of Registration': dd,
        'Date of Registration': d3,
        'Image Path': f'{path1}',
    }
    df = df.drop_duplicates(keep='first')
    df = df.append(data, ignore_index=True)
    df.to_csv(filename, index=False)
#     df = df.drop_duplicates()
else:
    data = {
        "Full Name": name,
        "Course": course,
        "Cohort": cohort,
        'Day of Registration': dd,
        'Date of Registration': d3,
        'Image Path': f'{path1}',
    }
    df = pd.DataFrame([data]).to_csv(filename, index=False)
#     df = df.drop_duplicates()
print('#'*10)
print('Database created')
print('Registration Done')
for image_path in list(paths.list_images('school_pic')):
    known_names = []
    known_encodings = []
    name = image_path.split('\\')[1]
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_img, model='hog')
    encodings = face_recognition.face_encodings(rgb_img, faces)
    for encoding in encodings:
        known_names.append(name)
        known_encodings.append(encoding)
data = {'Names': known_names, 'Encodings': known_encodings}
with open('school_att_encoder', 'wb') as f:
    f.write(pickle.dumps(data))
print('Good to go')


warnings.filterwarnings('ignore')

ws = tk.Tk()
ws.title('Attendance')
ws.geometry('400x100')
style = ttk.Style()
style.theme_use('classic')


data2 = pd.read_csv('Attendance3.csv')
name_data = data2['Full Name']
eve_filename = 'Evening Attendance.csv'
mon_filename = 'Morning Attendance.csv'
data = pd.read_csv('Profile.csv')
email_data = data['Email']
# name_data = data['First Name']
present = 'Present'
absent = 'Absent'
main = 'Main Attendance.csv'
dd = pd.Timestamp(datetime.today())
dd = dd.day_name()
today = date.today()
d3 = today.strftime("%m/%d/%y")

nu = Label(ws, text=' ', font=('Times', 15), fg='black', width=30)
nu.grid(row=0, column=3)
n = tk.StringVar(value="Select your Email")
emaily = ttk.Combobox(ws, width=30, height=20, textvariable=n)
emaily.grid(row=1, column=3, sticky="wesn")
with open('Profile.csv') as f:
    reader = csv.DictReader(f, delimiter=',')
    emaily['values'] = [row['Email'] for row in reader if row['Email'].strip()]

while True:
    now = datetime.now()
    mor_ses = '10:06:20 AM'
    mor_ses_h = mor_ses[0:2]
    mor_ses_m = mor_ses[3:5]
    mor_ses_s = mor_ses[6:8]
    mor_ses_p = mor_ses[9:].upper()
    mon = (mor_ses_p, mor_ses_h, mor_ses_m, mor_ses_s)
    mor_clo = '12:45:20 PM'
    mor_clo_h = mor_clo[0:2]
    mor_clo_m = mor_clo[3:5]
    mor_clo_s = mor_clo[6:8]
    mor_clo_p = mor_clo[9:].upper()
    mon_clo = (mor_clo_p, mor_clo_h, mor_clo_m, mor_clo_s)
    mon_filename = 'Morning_Attendance.csv'
    current_p = now.strftime("%p")
    current_h = now.strftime("%I")
    current_m = now.strftime("%M")
    current_s = now.strftime("%S")
    cur = (current_p, current_h, current_m, current_s)
    main_cur = (current_h, current_m, current_s, current_p)
    if cur == mon:
        messagebox.showinfo(ws, 'time to mark', icon='info')
        break
#         elif cur == mon_clo:
#             messagebox.showinfo(ws, 'attendance close',icon='info')
#             break


def sub_but():
    with open('school_att_encoder', 'rb') as f:
        saved_encodings = pickle.loads(f.read())
    nam = (emaily.get())
    for i in email_data:
        if i == nam:
            messagebox.showinfo(ws, f'You are welcome {nam}', icon='info')
            video = cv2.VideoCapture(0)
            while 1:
                cont = 1
                ret, frame = video.read()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = face_recognition.face_locations(rgb_frame, model='hog')
                video_encoding = face_recognition.face_encodings(
                    rgb_frame, face)
                recognized_faces = []
                cont = 1
                for encoding in video_encoding:
                    distance = face_recognition.face_distance(
                        encoding, saved_encodings['Encodings'])
                    min_distance = np.min(distance)
                    if min_distance > 0.5:
                        recognized_faces.append('Face not recognized')
                        break
                    else:
                        index = np.argmin(distance)
                        recognized_faces.append(
                            saved_encodings['Names'][index])
                        for (top, right, bottom, left), reg_name in zip(face, recognized_faces):
                            frame = cv2.rectangle(
                                frame, (left, top), (right, bottom), (126, 120, 186), 1)
                            frame = cv2.putText(
                                frame, reg_name, (left, top - 3), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 10, 125), 1)
                            im = reg_name
                            break
                        cont += 1
                        if (cont == 1):
                            break
                    for name in name_data:
                        if im == name:
                            break
                        else:
                            print('no')
                            break
                cv2.imshow('face_recognition', frame)
                key = cv2.waitKey(10)
                if key == ord('d'):
                    break
            video.release()
            cv2.destroyAllWindows()
            messagebox.showinfo(
                ws, f'{nam} your attendance has been marked for today {main_cur} as at {dd}, {d3}', icon='info')

    if os.path.exists(mon_filename):
        df = pd.read_csv(mon_filename)
        data = {
            'Full Name': name,
            f'{dd,d3}': f'{cur}'
        }
        df = df.append(data, ignore_index=True)
        df.to_csv(mon_filename, index=False)
        print(f'{name} your attendance has been marked for today {cur} as at {dd},{d3}')
    else:
        data = {
            'Full Name': name,
            f'{dd,d3}': f'{cur}'
        }
        df = pd.DataFrame([data]).to_csv(mon_filename, index=False)
        print(f'{name} your attendance has been marked for today {cur} as at {dd},{d3}')


submit2 = Button(ws, text='Mark', command=sub_but, font=('Times', 12))
submit2.grid(row=1, column=4)
while True:
    mor_clo = '12:32:20 PM'
    mor_clo_h = mor_clo[0:2]
    mor_clo_m = mor_clo[3:5]
    mor_clo_s = mor_clo[6:8]
    mor_clo_p = mor_clo[9:].upper()
    mon_clo = (mor_clo_p, mor_clo_h, mor_clo_m, mor_clo_s)
    current_p = now.strftime("%p")
    current_h = now.strftime("%I")
    current_m = now.strftime("%M")
    current_s = now.strftime("%S")
    cur = (current_p, current_h, current_m, current_s)
    if cur == mon_clo:
        messagebox.showinfo(ws, 'attendance close', icon='info')
        break

with open('school_att_encoder', 'rb') as f:
    saved_encodings = pickle.loads(f.read())
while True:

    nam = (emaily.get())
    for i in data:
        if i == nam:
            messagebox.showinfo(ws, f'You are welcome {nam}', icon='info')
        video = cv2.VideoCapture(0)
        while 1:
            ret, frame = video.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = face_recognition.face_locations(rgb_frame, model='hog')
            video_encoding = face_recognition.face_encodings(rgb_frame, face)
            recognized_faces = []
            cont = 1
            for encoding in video_encoding:
                distance = face_recognition.face_distance(
                    encoding, saved_encodings['Encodings'])
                min_distance = np.min(distance)
                if min_distance > 0.5:
                    recognized_faces.append('Face not recognized')
                    break
                else:
                    index = np.argmin(distance)
                    recognized_faces.append(saved_encodings['Names'][index])
                    for (top, right, bottom, left), reg_name in zip(face, recognized_faces):
                        frame = cv2.rectangle(
                            frame, (left, top), (right, bottom), (126, 120, 186), 1)
                        frame = cv2.putText(
                            frame, reg_name, (left, top - 3), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 10, 125), 1)

            cv2.imshow('face_recognition', frame)
            key = cv2.waitKey(10)
            if key == ord('d'):
                break
        video.release()
        cv2.destroyAllWindows()
    if os.path.exists(mon_filename):
        df = pd.read_csv(mon_filename)
        data = {
            'Full Name': nam,
            f'{dd,d3}': f'{cur}'
        }
        df = df.append(data, ignore_index=True)
        df.to_csv(mon_filename, index=False)
        df = df.drop_duplicates()
    else:
        data = {
            'Full Name': nam,
            f'{dd,d3}': f'{cur}'
        }
        df = pd.DataFrame([data]).to_csv(mon_filename, index=False)
        print(f'{nam} your attendance has been marked for today {cur} as at {dd},{d3}')

    for i in data:
        if i == nam:
            messagebox.showinfo(ws, f'You are welcome {nam}', icon='info')
            video = cv2.VideoCapture(0)
            while 1:
                ret, frame = video.read()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = face_recognition.face_locations(rgb_frame, model='hog')
                video_encoding = face_recognition.face_encodings(
                    rgb_frame, face)
                recognized_faces = []
                cont = 1
                for encoding in video_encoding:
                    distance = face_recognition.face_distance(
                        encoding, saved_encodings['Encodings'])
                    min_distance = np.min(distance)
                    if min_distance > 0.5:
                        recognized_faces.append('Face not recognized')
                        break
                    else:
                        index = np.argmin(distance)
                        recognized_faces.append(
                            saved_encodings['Names'][index])
                        for (top, right, bottom, left), reg_name in zip(face, recognized_faces):
                            frame = cv2.rectangle(
                                frame, (left, top), (right, bottom), (126, 120, 186), 1)
                            frame = cv2.putText(
                                frame, reg_name, (left, top - 3), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 10, 125), 1)
                            if (cont == 1):
                                break
                cv2.imshow('Attendance', frame)
                key = cv2.waitKey(10)
                if key == ord('d'):
                    break
            video.release()
            cv2.destroyAllWindows()
    if os.path.exists(eve_filename):
        df = pd.read_csv(eve_filename)
        data = {
            'Full Name': nam,
            f'{dd,d3}': f'{cur}'
        }
        df = df.append(data, ignore_index=True)
        df.to_csv(eve_filename, index=False)
        df = df.drop_duplicates()
    else:
        data = {
            'Full Name': nam,
            f'{dd,d3}': f'{cur}'
        }
        df = pd.DataFrame([data]).to_csv(eve_filename, index=False)
    if os.path.exists(main):
        df = pd.read_csv(main)
        data = {
            'Full Name': nam,
            f'{dd,d3}': present
        }
        df = df.append(data, ignore_index=True)
        df.to_csv(main, index=False)
        df = df.drop_duplicates()
    else:
        data = {
            'Full Name': nam,
            f'{dd,d3}': present
        }
        df = pd.DataFrame([data]).to_csv(main, index=False)
        cours = Label(profile, text=f'{nam} your attendance has been marked for closing today {now} as at {dd},{d3}', font=(
            'Times', 15),  fg='black')
        cours.grid(row=9, column=1)

#                 print(f'{nam} your attendance has been marked for closing today {now} as at {dd},{d3}')
ws.mainloop()
