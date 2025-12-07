import json
import pandas as pd
from functools import wraps
from flask_cors import CORS
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
import math

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"], supports_credentials=True)

AQI_CSV = "AQI_2020_2025.csv"
GAS_CSV = "Criteria_Gases_2020_2025.csv"
P10_CSV = "PM10_2020_2025.csv"
P25_CSV = "PM2.5_2020_2025.csv"
A1Y_CSV = "AQI_Prediction_1Y.csv"
A5Y_CSV = "AQI_Prediction_5Y.csv"


# -------------------------
#  STANDARD API RESPONSE
# -------------------------
def success_response(data, status_code=200):
    return jsonify({
        "status": "success",
        "data": data
    }), status_code


def error_response(message, status_code=500):
    return jsonify({
        "status": "error",
        "message": message
    }), status_code

def safe_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(round(value, 1))
    except:
        return None

def get_filters(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return error_response("Request body must be in JSON format.", 400)

        data = request.get_json()
        try:
            params = {
                "start_month": int(data.get("start_month")),
                "start_year": int(data.get("start_year")),
                "end_month": int(data.get("end_month")),
                "end_year": int(data.get("end_year")),
                "states": data.get("states", [])}
        except TypeError:
            return error_response("Date parameter must be filled with integer.", 400)

        return f(filters=params, *args, **kwargs)
    return decorated_function


def get_and_filter_data(filters: dict, csv: str | list[str], drop: str | list[str]) -> pd.DataFrame:
    start_month = filters["start_month"]
    start_year = filters["start_year"]
    end_month = filters["end_month"]
    end_year = filters["end_year"]
    states = filters["states"]

    if type(csv) == str:
        df = pd.read_csv(csv)
    else:
        df = pd.read_csv(csv[0])
        df.rename(columns={"Mean": "PM2.5 Mean"}, inplace=True)
        df = pd.concat([df, pd.read_csv(csv[1]).Mean], axis=1)
        df.rename(columns={"Mean": "PM10 Mean"}, inplace=True)

    df = df[
        (df.Month >= start_month)
        & (df.Month <= end_month)
        & (df.Year >= start_year)
        & (df.Year <= end_year)
        & (df["State Name"].isin(states))
    ]
    return df.drop(columns=drop)


def random_pie_chart_function(df: pd.DataFrame, filter: str) -> pd.DataFrame:
    if df.shape[0] > 5:
        sorted = df.sort_values(filter, ascending=False).iloc[:4]
        sorted.loc[-1] = [
            "Others",
            df.sort_values(filter, ascending=False).iloc[4:][filter].sum(),
        ]
    else:
        sorted = df.sort_values(filter, ascending=False)
    sorted.reset_index(drop=True, inplace=True)
    sorted[filter] = sorted[filter].round(1)
    return sorted


def change_month_type(df: pd.DataFrame) -> pd.DataFrame:
    df["dummy_date"] = pd.to_datetime(df.Month.astype(str) + "-01-2023")
    df.Month = df.dummy_date.dt.month_name()
    return df.drop(columns="dummy_date")

@app.route("/api/states", methods=["GET"])
def get_states():
    try:
        df = pd.read_csv(AQI_CSV)

        if "State Name" not in df.columns:
            return error_response("Column 'State Name' not found in CSV.", 500)

        states = sorted(df["State Name"].dropna().unique().tolist())

        return success_response( states)

    except Exception as e:
        return error_response(str(e))

@app.route("/api/sum_gases", methods=["POST"])
@get_filters
def get_sum_gases(filters: dict):
    try:
        df = get_and_filter_data(filters, GAS_CSV, ["Month", "Year", "State Name"])
        df = df.groupby("Parameter").sum()
        df = df[df.Mean == df.Mean.max()].reset_index()

        response_data = {
            "Name": str(df.Parameter.iloc[0]),
            "Sum": float(df.Mean.round(1).iloc[0]),
        }
        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/avg_aqi", methods=["POST"])
@get_filters
def get_avg_aqi(filters: dict):
    try:
        df = get_and_filter_data(
            filters, AQI_CSV, ["Month", "Year", "State Name", "Category"]
        )
        response_data = {"Mean": float(df.AQI.mean().round(1))}
        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/avg_pm25", methods=["POST"])
@get_filters
def get_avg_p25(filters: dict):
    try:
        df = get_and_filter_data(filters, P25_CSV, ["Month", "Year", "State Name"])
        response_data = {"Mean": float(df.Mean.mean().round(1))}
        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/avg_pm10", methods=["POST"])
@get_filters
def get_avg_p10(filters: dict):
    try:
        df = get_and_filter_data(filters, P10_CSV, ["Month", "Year", "State Name"])
        response_data = {"Mean": float(df.Mean.mean().round(1))}
        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/map_aqi", methods=["POST"])
@get_filters
def get_map_aqi(filters: dict):
    def categorize_aqi_value(aqi_val: float) -> str:
        if aqi_val <= 50:
            return "Good"
        elif 50 < aqi_val <= 100:
            return "Moderate"
        elif 100 < aqi_val <= 150:
            return "Unhealthy for Sensitive Groups"
        elif 150 < aqi_val <= 200:
            return "Unhealthy"
        elif 200 < aqi_val <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    try:
        df = get_and_filter_data(filters, AQI_CSV, ["Month", "Year", "Category"])
        df = df.groupby("State Name").mean().reset_index()
        df["Category"] = df.AQI.apply(categorize_aqi_value)

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {
                    "State Name": str(df.iloc[row, 0]),
                    "Mean AQI Value": float(df.iloc[row, 1]),
                    "AQI Category": str(df.iloc[row, 2]),
                }
            )

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/prec_gases", methods=["POST"])
@get_filters
def get_prec_gases(filters: dict):
    try:
        df = get_and_filter_data(filters, GAS_CSV, ["Month", "Year", "State Name"])
        df = df.groupby("Parameter").sum().reset_index()

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {"Name": str(df.iloc[row, 0]), "Total_Mass": float(df.iloc[row, 1])}
            )
            if row == 4:
                break

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/prec_aqi", methods=["POST"])
@get_filters
def get_prec_aqi(filters: dict):
    try:
        df = get_and_filter_data(filters, AQI_CSV, ["Month", "Year", "Category"])
        df = df.groupby("State Name").mean().reset_index()
        df = random_pie_chart_function(df, "AQI")

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {
                    "State": str(df.iloc[row, 0]),
                    "Mean": float(df.iloc[row, 1]),
                }
            )
            if row == 4:
                break

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/prec_p25", methods=["POST"])
@get_filters
def get_prec_p25(filters: dict):
    try:
        df = get_and_filter_data(filters, P25_CSV, ["Month", "Year"])
        df = df.groupby("State Name").sum().reset_index()
        df = random_pie_chart_function(df, "Mean")

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {
                    "State": str(df.iloc[row, 0]),
                    "Total_Mass": float(df.iloc[row, 1]),
                }
            )
            if row == 4:
                break

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/prec_p10", methods=["POST"])
@get_filters
def get_prec_p10(filters: dict):
    try:
        df = get_and_filter_data(filters, P10_CSV, ["Month", "Year"])
        df = df.groupby("State Name").sum().reset_index()
        df = random_pie_chart_function(df, "Mean")

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {
                    "State": str(df.iloc[row, 0]),
                    "Total_Mass": float(df.iloc[row, 1]),
                }
            )
            if row == 4:
                break

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/time_gases", methods=["POST"])
@get_filters
def get_time_gases(filters: dict):
    try:
        df = get_and_filter_data(filters, GAS_CSV, "State Name")
        df = df.groupby(["Year", "Month", "Parameter"]).mean().reset_index()
        df = df.pivot_table(
            index=["Year", "Month"], columns="Parameter", values="Mean"
        ).reset_index()
        df.columns.name = None
        df = change_month_type(df)

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {
                    "Date": f"{df.iloc[row, 1]} {df.iloc[row, 0]}",
                    "CO Mean": safe_float(df.iloc[row, 2]),
                    "NO2 Mean": safe_float(df.iloc[row, 3]),
                    "Ozone Mean": safe_float(df.iloc[row, 4]),
                    "SO2 Mean": safe_float(df.iloc[row, 5]),
                }
            )

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/time_pm", methods=["POST"])
@get_filters
def get_time_pm(filters: dict):
    try:
        df = get_and_filter_data(filters, [P25_CSV, P10_CSV], "State Name")
        df = df.groupby(["Year", "Month"])["AQI"].mean().reset_index()
        df = change_month_type(df)

        response_data = []
        for row in range(df.shape[0]):
            response_data.append(
                {
                    "Date": f"{df.iloc[row, 1]} {df.iloc[row, 0]}",
                    "PM2.5 Mean": float(df.iloc[row, 2].round(1)),
                    "PM10 Mean": float(df.iloc[row, 3].round(1)),
                }
            )

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/api/time_aqi", methods=["POST"])
def get_time_aqi():
    try:
        if not request.is_json:
            return error_response("Request body must be in JSON format.", 400)

        data = request.get_json()
        predict_type_raw = data.get("predict_type")
        try:
            predict_type = int(predict_type_raw) if predict_type_raw is not None else 1
        except TypeError:
            return error_response("Date parameter must be filled integer.", 400)

        df = pd.read_csv(AQI_CSV)
        if "AQI" in df.columns:
            df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
            df = df.dropna(subset=["AQI"])
        numeric_cols = df.select_dtypes(include=['number']).columns

        df = df.groupby(["Year", "Month"])[numeric_cols].mean().reset_index()   
        df = change_month_type(df)

        pred_df = pd.read_csv(A1Y_CSV if predict_type == 1 else A5Y_CSV)
        pred_df = change_month_type(pred_df)

        response_data = []
        for index, row in df.iterrows():
            response_data.append(
                {
                    "Date": f"{row['Month']} {row['Year']}",
                    "History": float(row["AQI"]),
                    "Predicted": None,
                    "CI_range": None,
                }
            )

        for index, row in pred_df.iterrows():
            if response_data and response_data[-1]["Predicted"] is None:
                response_data[-1]["Predicted"] = response_data[-1]["History"]
                response_data[-1]["CI_range"] = [
                    response_data[-1]["History"],
                    response_data[-1]["History"],
                ]

            response_data.append(
                {
                    "Date": f"{row['Month']} {row['Year']}",
                    "History": None,
                    "Predicted": float(row["Predicted_AQI"]),
                    "CI_range": [float(row["Lower_CI"]), float(row["Upper_CI"])],
                }
            )

        return success_response(response_data)

    except Exception as e:
        return error_response(str(e))



@app.route("/health", methods=["GET"])
def health():
    return success_response({"status": "ok"})



@app.route("/api/chatbot_aqi", methods=["POST"])
@get_filters
def chatbot_aqi(filters: dict):
    try:
        data = request.get_json()
        user_question = data.get(
            "question", "Berikan rekomendasi berdasarkan data ini."
        )

        df_history = get_and_filter_data(
            filters, AQI_CSV, ["Month", "Year", "Category"]
        )
        data_summary = df_history.describe().to_string()

        prompt = f"""
        Bertindaklah sebagai sistem rekomendasi kualitas udara.
        Berdasarkan data statistik AQI berikut:
        {data_summary}

        Pertanyaan User: {user_question}

        Tugas: Berikan 3-5 rekomendasi aksi konkret bagi pemangku kepentingan atau warga.
        
        PENTING: Keluaran HARUS berupa JSON Array murni tanpa format Markdown (```json).
        Struktur JSON wajib seperti ini:
        [
            {{
                "Title": "Judul Rekomendasi (Singkat & Padat)",
                "Description": "Penjelasan detail dan alasan berdasarkan data."
            }}
        ]
            """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()

        try:
            recommendations = json.loads(cleaned_text)
        except json.JSONDecodeError:
            recommendations = [
                {"Title": "Gagal Memformat Data", "Description": cleaned_text}
            ]

        recommendations = json.loads(cleaned_text)
        return (
            jsonify(
                {
                    "response": recommendations,
                    "context_used": "Data_AQI " + str(filters["states"]),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
