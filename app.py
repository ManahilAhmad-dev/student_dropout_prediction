
import gradio as gr
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("student_dropout_model.pkl")
model_features = joblib.load("model_features.pkl")


def predict_dropout(
    age,
    gender,
    family_income,
    internet_access,
    study_hours,
    attendance,
    assignment_delay,
    travel_time,
    part_time_job,
    scholarship,
    stress,
    gpa,
    semester_gpa,
    cgpa,
    semester,
    department,
    parental_education
):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Family_Income": [family_income],
        "Internet_Access": [internet_access],
        "Study_Hours_per_Day": [study_hours],
        "Attendance_Rate": [attendance],
        "Assignment_Delay_Days": [assignment_delay],
        "Travel_Time_Minutes": [travel_time],
        "Part_Time_Job": [part_time_job],
        "Scholarship": [scholarship],
        "Stress_Index": [stress],
        "GPA": [gpa],
        "Semester_GPA": [semester_gpa],
        "CGPA": [cgpa],
        "Semester": [semester],
        "Department": [department],
        "Parental_Education": [parental_education]
    })

    # Encode categorical variables
    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True,
        dtype=int
    )

    # Match model features
    input_encoded = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # Predict dropout probability
    probability = model.predict_proba(input_encoded)[0][1]

    # Determine risk category
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # Prediction
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        result = "Student may be at risk of dropout."
    else:
        result = "Student is predicted as unlikely to drop out."

    return (
        f"{probability * 100:.2f}%",
        risk,
        result
    )


# Create Gradio interface
demo = gr.Interface(
    fn=predict_dropout,

    inputs=[
        gr.Number(label="Age", value=20),
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(label="Family Income", value=50000),
        gr.Dropdown(["Yes", "No"], label="Internet Access"),
        gr.Number(label="Study Hours per Day", value=4),
        gr.Number(label="Attendance Rate (%)", value=75),
        gr.Number(label="Assignment Delay Days", value=2),
        gr.Number(label="Travel Time (Minutes)", value=30),
        gr.Dropdown(["Yes", "No"], label="Part-Time Job"),
        gr.Dropdown(["Yes", "No"], label="Scholarship"),
        gr.Number(label="Stress Index", value=5),
        gr.Number(label="GPA", value=2.5),
        gr.Number(label="Semester GPA", value=2.5),
        gr.Number(label="CGPA", value=2.5),
        gr.Number(label="Semester", value=5),

        gr.Dropdown(
            ["Arts", "Engineering", "CS", "Business", "Science"],
            label="Department"
        ),

        gr.Dropdown(
            ["High School", "Bachelor", "Master", "PhD"],
            label="Parental Education"
        )
    ],

    outputs=[
        gr.Textbox(label="Dropout Probability"),
        gr.Textbox(label="Risk Category"),
        gr.Textbox(label="Prediction")
    ],

    title="Student Dropout Risk Prediction",

    description=(
        "Enter student information to estimate dropout probability "
        "and risk level."
    )
)


demo.launch()
