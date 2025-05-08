import os
import json
import joblib
import pandas as pd
import streamlit as st

from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove whitespace from the top of the page and sidebar
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 3rem;
                    padding-left: 6vw;
                    padding-right: 6vw;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

st.subheader("Activity Monitoring - Aruba", divider="red")

wall_padding = 1
door_height = 22
table_width = 77
motion_circle_radius = 10
font_family = "Helvetica"
base_data_path = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(base_data_path, "time_series.csv"))
    df["sampled_time"] = pd.to_datetime(df["sampled_time"])

    df_ = pd.read_csv(os.path.join(base_data_path, "normalized_data.csv"))
    df_["sampled_time"] = df["sampled_time"]

    with open(os.path.join(base_data_path, "dataset_summary.json")) as f:
        summary = json.load(f)

    return df_, summary


normalized_data, summary = load_data()

st.markdown("<br/>", unsafe_allow_html=True)
activity_choice = st.pills(
    "Select a particular activity",
    list(summary["activity_encoder"].keys()),
    selection_mode="multi",
)


filtered_data = normalized_data.copy()
if activity_choice:
    encoded_activity_choices = [
        summary["activity_encoder"][activity] for activity in activity_choice
    ]

    filtered_data = filtered_data[
        filtered_data["activity_label"].isin(encoded_activity_choices)
    ]

st.markdown("<br/>", unsafe_allow_html=True)
user_time = st.select_slider(
    "Select a date and time",
    options=filtered_data["sampled_time"].dt.strftime("%d %b, %Y %H:%M:%S").tolist(),
)

data = filtered_data[filtered_data["sampled_time"] == user_time].to_dict(
    orient="records"
)


st.markdown("<br/>", unsafe_allow_html=True)
col1, _, col2 = st.columns([3, 1, 5])


with col1:
    if data:
        st.markdown("<br/>", unsafe_allow_html=True)

        sensors_triggered = set(
            x.split("-")[0]
            for x in data[0].keys()
            if (x.startswith("D") or x.startswith("T") or x.startswith("M"))
            and not pd.isna(data[0][x])
            and data[0][x] > 0
            and data[0][x] != "OFF"
            and data[0][x] != "CLOSED"
        )

        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            st.write(" Actual Activity: ")
            st.subheader(summary["activity_decoder"][str(data[0]["activity_label"])])

        with subcol2:
            st.write(" Predicted Activity: ")
            model = joblib.load(os.path.join(base_data_path, "model.pkl"))
            model_data = normalized_data.iloc[
                filtered_data[filtered_data["sampled_time"] == user_time].index[0], :
            ].to_list()[1:-1]
            prediction = model.predict([model_data])
            st.subheader(summary["activity_decoder"][str(prediction[0])])

        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.write("Sensors Triggered: ")
        st.write(sensors_triggered)


with col2:
    initial_drawing = {"objects": []}

    rooms = [
        {"name": "", "top": 0, "left": 0, "width": 700, "height": 500},
        {
            "name": "Kitchen",
            "top": 0,
            "left": 0,
            "width": 250 - wall_padding,
            "height": 150 - wall_padding,
        },
        {
            "name": "Dining",
            "top": 150,
            "left": 0,
            "width": 250 - wall_padding,
            "height": 150,
        },
        {
            "name": "Living",
            "top": 300,
            "left": 0,
            "width": 250 - wall_padding,
            "height": 200,
        },
        {
            "name": "Bathroom",
            "top": 0,
            "left": 250,
            "width": 150 - wall_padding,
            "height": 150 - wall_padding,
        },
        {
            "name": "Main Hall",
            "top": 150,
            "left": 250 - wall_padding,
            "width": 100,
            "height": 350,
        },
        {
            "name": "",
            "top": 0,
            "left": 400 - wall_padding,
            "width": 100 + wall_padding,
            "height": 250 - wall_padding,
        },
        {
            "name": "",
            "top": 150,
            "left": 350 - wall_padding,
            "width": 50,
            "height": 100 - wall_padding,
        },
        {
            "name": "Office",
            "top": 0,
            "left": 500,
            "width": 275,
            "height": 150 - wall_padding,
        },
        {
            "name": "Bedroom",
            "top": 150,
            "left": 500,
            "width": 275,
            "height": 100 - wall_padding,
        },
        {"name": "Bedroom", "top": 250, "left": 350, "width": 200, "height": 250},
        {
            "name": "Bathroom",
            "top": 250,
            "left": 550,
            "width": 150,
            "height": 125 - wall_padding,
        },
        {"name": "Closet", "top": 375, "left": 550, "width": 150, "height": 125},
    ]

    for room in rooms:
        rect = {
            "type": "rect",
            "left": room["left"],
            "top": room["top"],
            "width": room["width"],
            "height": room["height"],
            "fill": "rgba(173,216,230,0.4)",  # light blue
            "stroke": "black",
            "strokeWidth": 0,
            "name": room["name"],
        }
        initial_drawing["objects"].append(rect)

        if room["name"] != "":
            label = {
                "type": "textbox",
                "left": room["left"] + 5,
                "top": room["top"] + 5,
                "width": 100,
                "height": 100,
                "text": room["name"],
                "fontSize": 11,
                "fontFamily": font_family,
                "fill": "black",
            }
            initial_drawing["objects"].append(label)

    for door in [
        {
            "name": "D001",
            "top": 500 - door_height,
            "left": 250 - wall_padding,
            "width": 100,
            "height": door_height,
        },
        {
            "name": "D002",
            "top": 0,
            "left": 130 - wall_padding,
            "width": 100,
            "height": door_height,
        },
        {
            "name": "D003",
            "top": 150 - door_height,
            "left": 250 - wall_padding,
            "width": 100,
            "height": door_height,
        },
        {
            "name": "D004",
            "top": 0,
            "left": 400 - wall_padding,
            "width": 100,
            "height": door_height,
        },
    ]:
        rect = {
            "type": "rect",
            "left": door["left"],
            "top": door["top"],
            "width": door["width"],
            "height": door["height"],
            "fill": (
                "rgb(100,100,100)"
                if door["name"] not in sensors_triggered
                else "rgb(0, 128, 0)"
            ),
            "stroke": "black",
            "strokeWidth": 0,
            "name": door["name"],
        }
        initial_drawing["objects"].append(rect)

        label = {
            "type": "textbox",
            "left": door["left"] + 5,
            "top": door["top"] + 5,
            "width": 100,
            "height": 100,
            "text": door["name"],
            "fontSize": 10,
            "fontFamily": font_family,
            "fill": "white",
        }
        initial_drawing["objects"].append(label)

    for temperature in [
        {
            "name": "T001",
            "top": 350,
            "left": 380,
            "width": table_width,
            "height": door_height,
        },
        {
            "name": "T002",
            "top": 450,
            "left": 20,
            "width": table_width,
            "height": door_height,
        },
        {
            "name": "T003",
            "top": 110,
            "left": 20,
            "width": table_width,
            "height": door_height,
        },
        {
            "name": "T004",
            "top": 175,
            "left": 258,
            "width": table_width,
            "height": door_height,
        },
        {
            "name": "T005",
            "top": 65,
            "left": 610,
            "width": table_width,
            "height": door_height,
        },
    ]:
        react = {
            "type": "rect",
            "left": temperature["left"],
            "top": temperature["top"],
            "width": temperature["width"],
            "height": temperature["height"],
            "fill": (
                ""
                if temperature["name"] not in sensors_triggered
                else "rgba(255, 255, 255, 0.5)"
            ),
            "stroke": ("black"),
            "strokeWidth": 0,
            "name": temperature["name"],
        }
        initial_drawing["objects"].append(react)

        label = {
            "type": "textbox",
            "left": temperature["left"] + 5,
            "top": temperature["top"] + 5,
            "width": 100,
            "height": 100,
            "text": temperature["name"],
            "fontSize": 9,
            "fontFamily": font_family,
            "fill": (
                "rgb(105,105,105)"
                if temperature["name"] not in sensors_triggered
                else "black"
            ),
        }
        initial_drawing["objects"].append(label)

    for motion in [
        {"name": "M001", "top": 450, "left": 500},
        {"name": "M002", "top": 450, "left": 380},
        {"name": "M003", "top": 390, "left": 380},
        {"name": "M004", "top": 280, "left": 530},
        {"name": "M005", "top": 280, "left": 440},
        {"name": "M006", "top": 280, "left": 360},
        {"name": "M007", "top": 330, "left": 430, "area": "large"},
        {"name": "M008", "top": 280, "left": 320},
        {"name": "M009", "top": 300, "left": 220},
        {"name": "M010", "top": 370, "left": 220},
        {"name": "M011", "top": 430, "left": 290},
        {"name": "M012", "top": 380, "left": 120},
        {"name": "M013", "top": 280, "left": 120},
        {"name": "M014", "top": 170, "left": 120},
        {"name": "M015", "top": 70, "left": 60},
        {"name": "M016", "top": 30, "left": 160},
        {"name": "M017", "top": 70, "left": 160},
        {"name": "M018", "top": 140, "left": 200},
        {"name": "M019", "top": 40, "left": 70, "area": "large"},
        {"name": "M020", "top": 280, "left": 120, "area": "large"},
        {"name": "M021", "top": 210, "left": 280},
        {"name": "M022", "top": 190, "left": 380},
        {"name": "M023", "top": 170, "left": 450},
        {"name": "M024", "top": 150, "left": 550, "area": "large"},
        {"name": "M025", "top": 100, "left": 650},
        {"name": "M026", "top": 20, "left": 590},
        {"name": "M027", "top": 30, "left": 520, "area": "large"},
        {"name": "M028", "top": 100, "left": 450},
        {"name": "M029", "top": 50, "left": 370},
        {"name": "M030", "top": 30, "left": 420},
        {"name": "M031", "top": 80, "left": 280},
    ]:
        circle = {
            "type": "circle",
            "left": motion["left"],
            "top": motion["top"],
            "radius": (
                0
                if motion["name"] not in sensors_triggered
                else (
                    motion_circle_radius
                    if "area" not in motion
                    else motion_circle_radius * 5
                )
            ),
            "fill": (
                "rgba(255, 0, 0, 0.5)"
                if "area" not in motion
                else "rgba(255, 0, 0, 0.15)"
            ),
            "stroke": "black",
            "strokeWidth": 0,
            "name": motion["name"],
        }
        initial_drawing["objects"].append(circle)

        label = {
            "type": "textbox",
            "left": (
                (motion["left"] - 2) if "area" not in motion else (motion["left"] + 38)
            ),
            "top": (
                (motion["top"] + 25) if "area" not in motion else (motion["top"] + 45)
            ),
            "width": 100,
            "height": 100,
            "text": motion["name"],
            "fontSize": 9,
            "fontFamily": font_family,
            "fill": (
                "rgb(105,105,105)"
                if motion["name"] not in sensors_triggered
                else "black"
            ),
        }
        initial_drawing["objects"].append(label)

    # Create the canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=1,
        width=700,
        height=500,
        initial_drawing=initial_drawing,
        update_streamlit=True,
        drawing_mode="point",
        key="canvas",
    )
