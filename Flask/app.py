from flask import Flask, render_template, url_for, request, Response
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import io
import urllib
import base64

app = Flask(__name__)


@app.route("/", methods = ['POST', 'GET'])
def index():

    return render_template("home.html")

@app.route('/form')
def form():
    return render_template('inputs.html')

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        key1 = list(form_data.keys())[0]
        key2 = list(form_data.keys())[1]
        key3 = list(form_data.keys())[2]
        value1 = list(form_data.values())[0]
        value2 = list(form_data.values())[1]
        value3 = list(form_data.values())[2]
        return render_template('data.html',key1=key1, key2=key2, key3=key3, value1=value1, value2=value2, value3=value3)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            npimg = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_face_mesh = mp.solutions.face_mesh
            face_detection = mp_face_mesh.FaceMesh(static_image_mode=True)
            results = face_detection.process(img_rgb)
            if results.multi_face_landmarks:
                face_coordinates = results.multi_face_landmarks[0]
                h = img.shape[0]
                w = img.shape[1]
                x_array = []
                y_array = []
                z_array = []
                for i in face_coordinates.landmark:
                    x_array.append(i.x)
                    y_array.append(i.y)
                    z_array.append(i.z)

                int_z = np.array(z_array)
                normalized_array = (int_z - min(int_z)) / (max(int_z) - min(int_z)) * 200
                x_array = np.array(x_array)*w
                y_array = np.array(y_array)*h
                int_x = x_array.astype(int)
                int_y = y_array.astype(int)
                coords = list(zip(int_x, int_y))

                        # Create a scatter plot
                plt.figure(figsize=(10, 10))
                blank = img_rgb
                for idx, i in enumerate(blank):
                    blank[idx] = [0,0,0]
                x_y = []
                face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
                face_oval_list = list(face_oval)
                for i, j in face_oval_list:
                    x_y.append(coords[i])  # Get the coordinates for point i
                    x_y.append(coords[j]) # Get the coordinates for point j
                transpose = np.array(x_y).T
                max_x = max(transpose[0]) 
                min_x = min(transpose[0])
                max_y = max(transpose[1]) 
                min_y = min(transpose[1])
                border_res = 50
                plt.xlim(max_x + border_res, min_x - border_res)
                plt.ylim(max_y + border_res, min_y - border_res)
                plt.imshow(blank, cmap='gray')
                plt.axis("off")
                # for index, (x, y) in enumerate(coords):

                plt.scatter(int_x, int_y, marker='o', label='Point', c=normalized_array, s=1, cmap='viridis', vmin=0, vmax=55)            
                plt.show()
            
                    # Convert the plot to a PNG image
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)

                # Convert the PNG image to a base64 encoded string
                plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())
                plt.clf()

                return render_template("example_temp.html", plot_url=plot_url)
            else:
                string = "Face cant be detected, reload page and show more of facial features"
                return render_template("face_undetected_error_handling.html", string=string)
          
    return render_template('upload.html')
    
def generate_frames():
    camera = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)
            if results.multi_face_landmarks:
                # There is a prediction, so display the frame with landmarks
                face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
                face_oval_list = list(face_oval)
                face_coordinates = results.multi_face_landmarks[0]
                h = frame.shape[0]
                w = frame.shape[1]

                x_array = []
                y_array = []
                z_array = []
                for i in face_coordinates.landmark:
                    x_array.append(i.x)
                    y_array.append(i.y)
                    z_array.append(i.z)
                int_z = np.array(z_array)
                normalized_array = (int_z - min(int_z)) / (max(int_z) - min(int_z)) * 200
                x_array = np.array(x_array) * w
                y_array = np.array(y_array) * h
                int_x = x_array.astype(int)
                int_y = y_array.astype(int)
                coords = list(zip(int_x, int_y))
                blank = frame
                for idx, i in enumerate(blank):
                    blank[idx] = [0,0,0]
                for index, (x, y) in enumerate(coords):
                    value = int(normalized_array[index] + 55)
                    cv2.circle(blank, (x, y), 1, (0, value, 0), -1)  # Draw a small red circle at the point
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route("/show_leaderboard_plot")
def plot_leaders():
    with open("mock_leaderboard.json") as jsonfile:
        data = json.load(jsonfile)

    users = data.keys()
    points = []
    for i in users:
        points.append(data[i]["points"])
    leaderboard_df = pd.DataFrame({"Users": users, "points": points})

    plt.bar(leaderboard_df["Users"], leaderboard_df["points"])

    # Convert the plot to a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the PNG image to a base64 encoded string
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

    plt.clf()
    return render_template('leaderboard.html', plot_url=plot_url)

@app.route("/show_leaderboard")
def tab_leaders():
    with open("mock_leaderboard.json") as jsonfile:
        data = json.load(jsonfile)

    users = data.keys()
    points = []
    for i in users:
        points.append(data[i]["points"])
    leaderboard_df = pd.DataFrame({"Users": users, "points": points})
  
    leaderboard_df_sorted = leaderboard_df.sort_values("points", ignore_index=True, ascending=False)
    leaderboard_df_sorted.index.name = "Ranking"

    # Convert the DataFrame to HTML
    leaderboard_html = leaderboard_df_sorted.to_html()

    return render_template('leaderboard_tab.html', leaderboard=leaderboard_html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_feed_link')
def live_feed():
    return render_template('show_live_feed.html')

if __name__ == "__main__":
    app.run(debug=True)