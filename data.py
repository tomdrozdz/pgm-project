import pandas as pd


def load_data(path="Occupancy_Estimation.csv") -> pd.DataFrame:
    data = pd.read_csv(
        path,
        names=[
            "Date",
            "Time",
            "Temp1",
            "Temp2",
            "Temp3",
            "Temp4",
            "Light1",
            "Light2",
            "Light3",
            "Light4",
            "Sound1",
            "Sound2",
            "Sound3",
            "Sound4",
            "CO2",
            "CO2_Slope",
            "Motion1",
            "Motion2",
            "Occupancy",
        ],
        header=0,
    )

    data["Timestamp"] = pd.to_datetime(data["Date"] + " " + data["Time"])
    data.drop(columns=["Date", "Time"], inplace=True)

    return data
