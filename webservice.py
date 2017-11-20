from flask import Flask, jsonify, request
from sklearn.externals import joblib
import peewee
import pandas as pd
import numpy as np
import pickle

# instancia do Flask
app = Flask(__name__)

def normalize(col, min, max):
    return (col.subtract(min)).divide(max)

def getEuclidianDistance(col1x, col1y, col2x, col2y):
    print col1x
    return np.sqrt(col2x.subtract(col1x).pow(2) + col2y.subtract(col1y).pow(2))

def normalizeDF(df):
    df = df.drop(["pupil_left_x", "pupil_right_x", "pupil_left_y", "pupil_right_y"],axis=1)

    df["nose_tip_x"] = normalize(df["nose_tip_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["mouth_left_x"] = normalize(df["mouth_left_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["mouth_right_x"] = normalize(df["mouth_right_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eyebrow_left_outer_x"] = normalize(df["eyebrow_left_outer_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eyebrow_left_inner_x"] = normalize(df["eyebrow_left_inner_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_left_outer_x"] = normalize(df["eye_left_outer_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_left_top_x"] = normalize(df["eye_left_top_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_left_bottom_x"] = normalize(df["eye_left_bottom_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_left_inner_x"] = normalize(df["eye_left_inner_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eyebrow_right_inner_x"] = normalize(df["eyebrow_right_inner_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eyebrow_right_outer_x"] = normalize(df["eyebrow_right_outer_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_right_inner_x"] = normalize(df["eye_right_inner_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_right_top_x"] = normalize(df["eye_right_top_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_right_bottom_x"] = normalize(df["eye_right_bottom_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["eye_right_outer_x"] = normalize(df["eye_right_outer_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["nose_left_x"] = normalize(df["nose_left_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["nose_right_x"] = normalize(df["nose_right_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["nose_left_alar_top_x"] = normalize(df["nose_left_alar_top_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["nose_right_alar_top_x"] = normalize(df["nose_right_alar_top_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["nose_left_alar_out_tip_x"] = normalize(df["nose_left_alar_out_tip_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["nose_right_alar_out_tip_x"] = normalize(df["nose_right_alar_out_tip_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["upper_lip_top_x"] = normalize(df["upper_lip_top_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["upper_lip_bottom_x"] = normalize(df["upper_lip_bottom_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["under_lip_top_x"] = normalize(df["under_lip_top_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    df["under_lip_bottom_x"] = normalize(df["under_lip_bottom_x"], df["face_rectangle_left"], df["face_rectangle_width"])
    
    df["nose_tip_y"] = normalize(df["nose_tip_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["mouth_left_y"] = normalize(df["mouth_left_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["mouth_right_y"] = normalize(df["mouth_right_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eyebrow_left_outer_y"] = normalize(df["eyebrow_left_outer_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eyebrow_left_inner_y"] = normalize(df["eyebrow_left_inner_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_left_outer_y"] = normalize(df["eye_left_outer_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_left_top_y"] = normalize(df["eye_left_top_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_left_bottom_y"] = normalize(df["eye_left_bottom_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_left_inner_y"] = normalize(df["eye_left_inner_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eyebrow_right_inner_y"] = normalize(df["eyebrow_right_inner_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eyebrow_right_outer_y"] = normalize(df["eyebrow_right_outer_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_right_inner_y"] = normalize(df["eye_right_inner_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_right_top_y"] = normalize(df["eye_right_top_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_right_bottom_y"] = normalize(df["eye_right_bottom_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["eye_right_outer_y"] = normalize(df["eye_right_outer_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["nose_left_y"] = normalize(df["nose_left_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["nose_right_y"] = normalize(df["nose_right_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["nose_left_alar_top_y"] = normalize(df["nose_left_alar_top_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["nose_right_alar_top_y"] = normalize(df["nose_right_alar_top_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["nose_left_alar_out_tip_y"] = normalize(df["nose_left_alar_out_tip_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["nose_right_alar_out_tip_y"] = normalize(df["nose_right_alar_out_tip_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["upper_lip_top_y"] = normalize(df["upper_lip_top_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["upper_lip_bottom_y"] = normalize(df["upper_lip_bottom_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["under_lip_top_y"] = normalize(df["under_lip_top_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    df["under_lip_bottom_y"] = normalize(df["under_lip_bottom_y"], df["face_rectangle_top"], df["face_rectangle_height"])
    
    df["nose_left_top"] = getEuclidianDistance(df["nose_left_alar_top_x"], df["nose_left_alar_top_y"], df["nose_left_alar_out_tip_x"], df["nose_left_alar_out_tip_y"])  
    df["nose_right_top"] = getEuclidianDistance(df["nose_right_alar_top_x"], df["nose_right_alar_top_y"], df["nose_right_alar_out_tip_x"], df["nose_right_alar_out_tip_y"])        
    df["nose_left_bottom"] = getEuclidianDistance(df["nose_left_x"], df["nose_left_y"], df["nose_left_alar_top_x"], df["nose_left_alar_top_y"])  
    df["nose_right_bottom"] = getEuclidianDistance(df["nose_right_x"], df["nose_right_y"], df["nose_right_alar_top_x"], df["nose_right_alar_top_y"])    
    
    df["eye_triangle_left"] = getEuclidianDistance(df["eye_left_inner_x"], df["eye_left_inner_y"], df["nose_tip_x"], df["nose_tip_y"])    
    df["eye_triangle_right"] = getEuclidianDistance(df["eye_right_inner_x"], df["eye_right_inner_y"], df["nose_tip_x"], df["nose_tip_y"])    
    df["mouth_triangle_left"] = getEuclidianDistance(df["mouth_left_x"], df["mouth_left_y"], df["nose_tip_x"], df["nose_tip_y"])    
    df["mouth_triangle_right"] = getEuclidianDistance(df["mouth_right_x"], df["mouth_right_y"], df["nose_tip_x"], df["nose_tip_y"])    
    df["mouth_size"] = getEuclidianDistance(df["mouth_left_x"], df["mouth_left_y"], df["mouth_right_x"], df["mouth_right_y"])
    df["eyebrow_left_size"] = getEuclidianDistance(df["eyebrow_left_inner_x"], df["eyebrow_left_inner_y"], df["eyebrow_left_outer_x"], df["eyebrow_left_outer_y"])
    df["eyebrow_right_size"] = getEuclidianDistance(df["eyebrow_right_inner_x"], df["eyebrow_right_inner_y"], df["eyebrow_right_outer_x"], df["eyebrow_right_outer_y"])
    df["eye_left_x_size"] = getEuclidianDistance(df["eye_left_inner_x"], df["eye_left_inner_y"], df["eye_left_outer_x"], df["eye_left_outer_y"])
    df["eye_right_x_size"] = getEuclidianDistance(df["eye_right_inner_x"], df["eye_right_inner_y"], df["eye_right_outer_x"], df["eye_right_outer_y"])
    df["eye_left_y_size"] = getEuclidianDistance(df["eye_left_top_x"], df["eye_left_top_y"], df["eye_left_bottom_x"], df["eye_left_bottom_y"])
    df["eye_right_y_size"] = getEuclidianDistance(df["eye_right_top_x"], df["eye_right_top_y"], df["eye_right_bottom_x"], df["eye_right_bottom_y"])
    df["nose_tip_size"] = getEuclidianDistance(df["nose_left_x"], df["nose_left_y"], df["nose_right_x"], df["nose_right_y"])
    df["nose_left_alar_size"] = getEuclidianDistance(df["nose_left_alar_top_x"], df["nose_left_alar_top_y"], df["nose_right_alar_top_x"], df["nose_right_alar_top_y"])
    df["nose_left_alar_tip_size"] = getEuclidianDistance(df["nose_left_alar_out_tip_x"], df["nose_left_alar_out_tip_y"], df["nose_right_alar_out_tip_x"], df["nose_right_alar_out_tip_y"])
    df["upper_lip_size"] = getEuclidianDistance(df["upper_lip_top_x"], df["upper_lip_top_y"], df["upper_lip_bottom_x"], df["upper_lip_bottom_y"])
    df["under_lip_size"] = getEuclidianDistance(df["under_lip_top_x"], df["under_lip_top_y"], df["under_lip_bottom_x"], df["under_lip_bottom_y"])

    df = df.drop(["face_rectangle_top", "face_rectangle_left", "face_rectangle_width", "face_rectangle_height"],axis=1)
    df = df.drop(["mouth_left_x", "mouth_left_y", "mouth_right_x", "mouth_right_y"],axis=1)
    df = df.drop(["eyebrow_left_inner_x", "eyebrow_left_inner_y", "eyebrow_left_outer_x", "eyebrow_left_outer_y"],axis=1)
    df = df.drop(["eyebrow_right_inner_x", "eyebrow_right_inner_y", "eyebrow_right_outer_x", "eyebrow_right_outer_y"],axis=1)
    df = df.drop(["eye_left_inner_x", "eye_left_inner_y", "eye_left_outer_x", "eye_left_outer_y"],axis=1)
    df = df.drop(["eye_right_inner_x", "eye_right_inner_y", "eye_right_outer_x", "eye_right_outer_y"],axis=1)
    df = df.drop(["eye_left_top_x", "eye_left_top_y", "eye_left_bottom_x", "eye_left_bottom_y"],axis=1)    
    df = df.drop(["eye_right_top_x", "eye_right_top_y", "eye_right_bottom_x", "eye_right_bottom_y"],axis=1)
    df = df.drop(["nose_left_x", "nose_left_y", "nose_right_x", "nose_right_y"],axis=1)
    df = df.drop(["nose_left_alar_top_x", "nose_left_alar_top_y", "nose_right_alar_top_x", "nose_right_alar_top_y"],axis=1)
    df = df.drop(["nose_left_alar_out_tip_x", "nose_left_alar_out_tip_y", "nose_right_alar_out_tip_x", "nose_right_alar_out_tip_y"],axis=1)
    df = df.drop(["upper_lip_top_x", "upper_lip_top_y", "upper_lip_bottom_x", "upper_lip_bottom_y"],axis=1)
    df = df.drop(["under_lip_top_x", "under_lip_top_y", "under_lip_bottom_x", "under_lip_bottom_y"],axis=1)
    
    return df

# POST /api/face
@app.route('/api/face', methods=['POST'])
def check_face():

    dados = request.json

    columns = ["face_id","face_rectangle_top", "face_rectangle_left", "face_rectangle_width", "face_rectangle_height", "pupil_left_x", "pupil_left_y", "pupil_right_x", "pupil_right_y", "nose_tip_x", "nose_tip_y", "mouth_left_x", "mouth_left_y", "mouth_right_x", "mouth_right_y", "eyebrow_left_outer_x", "eyebrow_left_outer_y", "eyebrow_left_inner_x", "eyebrow_left_inner_y", "eye_left_outer_x", "eye_left_outer_y", "eye_left_top_x", "eye_left_top_y", "eye_left_bottom_x", "eye_left_bottom_y", "eye_left_inner_x", "eye_left_inner_y", "eyebrow_right_inner_x", "eyebrow_right_inner_y", "eyebrow_right_outer_x", "eyebrow_right_outer_y",
               "eye_right_inner_x", "eye_right_inner_y", "eye_right_top_x", "eye_right_top_y", "eye_right_bottom_x", "eye_right_bottom_y", "eye_right_outer_x", "eye_right_outer_y", "nose_left_x", "nose_left_y", "nose_right_x", "nose_right_y", "nose_left_alar_top_x", "nose_left_alar_top_y", "nose_right_alar_top_x", "nose_right_alar_top_y", "nose_left_alar_out_tip_x", "nose_left_alar_out_tip_y", "nose_right_alar_out_tip_x", "nose_right_alar_out_tip_y", "upper_lip_top_x", "upper_lip_top_y", "upper_lip_bottom_x", "upper_lip_bottom_y", "under_lip_top_x", "under_lip_top_y", "under_lip_bottom_x", "under_lip_bottom_y", "gender"]

    df_dados = [dados["faceId"],dados["faceRectangle"]["top"],dados["faceRectangle"]["left"],dados["faceRectangle"]["width"],dados["faceRectangle"]["height"],dados["faceLandmarks"]["pupilLeft"]["x"],dados["faceLandmarks"]["pupilLeft"]["y"],dados["faceLandmarks"]["pupilRight"]["x"],dados["faceLandmarks"]["pupilRight"]["y"],dados["faceLandmarks"]["noseTip"]["x"],dados["faceLandmarks"]["noseTip"]["y"],dados["faceLandmarks"]["mouthLeft"]["x"],dados["faceLandmarks"]["mouthLeft"]["y"],dados["faceLandmarks"]["mouthRight"]["x"],dados["faceLandmarks"]["mouthRight"]["x"],dados["faceLandmarks"]["eyebrowLeftOuter"]["x"],dados["faceLandmarks"]["eyebrowLeftOuter"]["y"],dados["faceLandmarks"]["eyebrowLeftInner"]["x"],dados["faceLandmarks"]["eyebrowLeftInner"]["y"],dados["faceLandmarks"]["eyeLeftOuter"]["x"],dados["faceLandmarks"]["eyeLeftOuter"]["y"],dados["faceLandmarks"]["eyeLeftTop"]["x"],dados["faceLandmarks"]["eyeLeftTop"]["y"],dados["faceLandmarks"]["eyeLeftBottom"]["x"],dados["faceLandmarks"]["eyeLeftBottom"]["y"],dados["faceLandmarks"]["eyeLeftInner"]["x"],dados["faceLandmarks"]["eyeLeftInner"]["y"],dados["faceLandmarks"]["eyebrowRightInner"]["x"],dados["faceLandmarks"]["eyebrowRightInner"]["y"],dados["faceLandmarks"]["eyebrowRightOuter"]["x"],dados["faceLandmarks"]["eyebrowRightOuter"]["y"],dados["faceLandmarks"]["eyeRightInner"]["x"],dados["faceLandmarks"]["eyeRightInner"]["y"],dados["faceLandmarks"]["eyeRightTop"]["x"],dados["faceLandmarks"]["eyeRightTop"]["y"],dados["faceLandmarks"]["eyeRightBottom"]["x"],dados["faceLandmarks"]["eyeRightBottom"]["y"],dados["faceLandmarks"]["eyeRightOuter"]["x"],dados["faceLandmarks"]["eyeRightOuter"]["y"],dados["faceLandmarks"]["noseRootLeft"]["x"],dados["faceLandmarks"]["noseRootLeft"]["y"],dados["faceLandmarks"]["noseRootRight"]["x"],dados["faceLandmarks"]["noseRootRight"]["y"],dados["faceLandmarks"]["noseLeftAlarTop"]["x"],dados["faceLandmarks"]["noseLeftAlarTop"]["y"],dados["faceLandmarks"]["noseRightAlarTop"]["x"],dados["faceLandmarks"]["noseRightAlarTop"]["y"],dados["faceLandmarks"]["noseLeftAlarOutTip"]["x"],dados["faceLandmarks"]["noseLeftAlarOutTip"]["y"],dados["faceLandmarks"]["noseRightAlarOutTip"]["x"],dados["faceLandmarks"]["noseRightAlarOutTip"]["y"],dados["faceLandmarks"]["upperLipTop"]["x"],dados["faceLandmarks"]["upperLipTop"]["y"],dados["faceLandmarks"]["upperLipBottom"]["x"],dados["faceLandmarks"]["upperLipBottom"]["y"],dados["faceLandmarks"]["underLipTop"]["x"],dados["faceLandmarks"]["underLipTop"]["y"],dados["faceLandmarks"]["underLipBottom"]["x"],dados["faceLandmarks"]["underLipBottom"]["y"], dados["faceAttributes"]["gender"]]

    df = pd.DataFrame(columns=columns)
    df.loc[0] = df_dados
    df.to_csv("temp.csv", sep=',', index=False)

    df = pd.read_csv("./temp.csv")
    df = normalizeDF(df)
    df = df.drop(["face_id", "gender", "nose_tip_x", "nose_tip_y"],axis=1)
    df.to_csv("temp.csv", sep=',', index=False)
    print df
    
    ada = joblib.load('model.pkl')
    y_pred_ada = ada.predict(df)
    print(y_pred_ada)

    if(y_pred_ada[0] == 1):
        auth = True
    else:
        auth = False

    return jsonify({'status': 200, 'auth': auth})


if __name__ == '__main__':
    app.run(debug=True)
