import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import pickle
filename = './model/customer_churn.model'
model = pickle.load(open(filename, 'rb'))
# Dummy model for illustration; replace this with your actual model
# class DummyModel:
#     def predict(self, X):
#         return np.random.choice([0, 1], size=len(X))

# model = DummyModel()



# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Telecommunication Customer Churn Prediction"

# Define the app layout with enhanced styling
app.layout = html.Div(
    style={
        "display": "flex", 
        "flexDirection": "column", 
        "alignItems": "center", 
        "justifyContent": "center", 
        "minHeight": "100vh", 
        "backgroundColor": "#f0f2f5", 
        "padding": "20px"
    },
    children=[
        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "padding": "30px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "maxWidth": "600px",
                "width": "100%",
                "boxSizing": "border-box"
            },
            children=[
                html.H1(
                    "Customer Churn Prediction",
                    style={"textAlign": "center", "color": "#4A90E2", "marginBottom": "20px"}
                ),
                
                html.P(
                    "Enter customer details to predict the likelihood of churn.",
                    style={"textAlign": "center", "color": "#6b6b6b", "fontSize": "16px"}
                ),
                
                # Container for form fields with column layout and pastel background
                html.Div(
                    style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "backgroundColor": "#e0f7fa", "padding": "15px", "borderRadius": "8px"},
                    children=[
                        html.Div([
                            html.Label("Gender"),
                            dcc.Dropdown(id="gender", options=[{"label": "Male", "value": "Male"}, {"label": "Female", "value": "Female"}], placeholder="Select Gender")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Senior Citizen"),
                            dcc.Dropdown(id="SeniorCitizen", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}], placeholder="Is Senior Citizen?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Partner"),
                            dcc.Dropdown(id="Partner", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}], placeholder="Has Partner?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Dependents"),
                            dcc.Dropdown(id="Dependents", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}], placeholder="Has Dependents?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Phone Service"),
                            dcc.Dropdown(id="PhoneService", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}], placeholder="Has Phone Service?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Multiple Lines"),
                            dcc.Dropdown(id="MultipleLines", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Phone Service", "value": "No phone service"}], placeholder="Has Multiple Lines?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Internet Service"),
                            dcc.Dropdown(id="InternetService", options=[{"label": "DSL", "value": "DSL"}, {"label": "Fiber optic", "value": "Fiber optic"}, {"label": "No", "value": "No"}], placeholder="Internet Service Type")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Online Security"),
                            dcc.Dropdown(id="OnlineSecurity", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Internet Service", "value": "No internet service"}], placeholder="Has Online Security?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Online Backup"),
                            dcc.Dropdown(id="OnlineBackup", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Internet Service", "value": "No internet service"}], placeholder="Has Online Backup?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Device Protection"),
                            dcc.Dropdown(id="DeviceProtection", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Internet Service", "value": "No internet service"}], placeholder="Has Device Protection?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Tech Support"),
                            dcc.Dropdown(id="TechSupport", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Internet Service", "value": "No internet service"}], placeholder="Has Tech Support?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Streaming TV"),
                            dcc.Dropdown(id="StreamingTV", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Internet Service", "value": "No internet service"}], placeholder="Has Streaming TV?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Streaming Movies"),
                            dcc.Dropdown(id="StreamingMovies", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No Internet Service", "value": "No internet service"}], placeholder="Has Streaming Movies?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Contract"),
                            dcc.Dropdown(id="Contract", options=[{"label": "Month-to-month", "value": "Month-to-month"}, {"label": "One year", "value": "One year"}, {"label": "Two year", "value": "Two year"}], placeholder="Contract Type")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Paperless Billing"),
                            dcc.Dropdown(id="PaperlessBilling", options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}], placeholder="Has Paperless Billing?")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Payment Method"),
                            dcc.Dropdown(id="PaymentMethod", options=[{"label": "Electronic check", "value": "Electronic check"}, {"label": "Mailed check", "value": "Mailed check"}, {"label": "Bank transfer (automatic)", "value": "Bank transfer (automatic)"}, {"label": "Credit card (automatic)", "value": "Credit card (automatic)"}], placeholder="Payment Method")
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Monthly Charges"),
                            dcc.Input(id="MonthlyCharges", type="number", min=0,placeholder="Monthly Charges", style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #ddd"})
                        ], style={"flex": "1 1 45%", "minWidth": "250px"}),

                        html.Div([
                            html.Label("Total Charges"),
                            dcc.Input(id="TotalCharges", type="number", min=0, placeholder="Total Charges", style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #ddd"})
                        ], style={"flex": "1 1 45%", "minWidth": "250px"})
                    ],
                ),
                
                html.Button(
                    "Predict Churn", 
                    id="predict-button", 
                    n_clicks=0, 
                    style={
                        "backgroundColor": "#4A90E2", 
                        "color": "white", 
                        "padding": "12px", 
                        "border": "none", 
                        "borderRadius": "8px", 
                        "width": "100%", 
                        "marginTop": "20px",
                        "fontWeight": "bold",
                        "fontSize": "16px",
                        "cursor": "pointer"
                    }
                ),
                
                html.Div(id="prediction-result", style={"marginTop": "20px", "textAlign": "center", "fontSize": "20px", "color": "#333"}),
            ]
        )
    ]
)

# Define the callback to make predictions
@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    [Input("gender", "value"), Input("SeniorCitizen", "value"), Input("Partner", "value"),
     Input("Dependents", "value"), Input("PhoneService", "value"), Input("MultipleLines", "value"),
     Input("InternetService", "value"), Input("OnlineSecurity", "value"), Input("OnlineBackup", "value"),
     Input("DeviceProtection", "value"), Input("TechSupport", "value"), Input("StreamingTV", "value"),
     Input("StreamingMovies", "value"), Input("Contract", "value"), Input("PaperlessBilling", "value"),
     Input("PaymentMethod", "value"), Input("MonthlyCharges", "value"), Input("TotalCharges", "value")]
)
def predict_churn(n_clicks, *inputs):
    if n_clicks > 0:
        monthly_charges = inputs[-2]
        total_charges = inputs[-1]
        
        if (monthly_charges is None or monthly_charges <= 0 or not isinstance(monthly_charges, (int, float))) or \
           (total_charges is None or total_charges <= 0 or not isinstance(total_charges, (int, float))):
            return "Error: Monthly Charges and Total Charges must be positive numbers."
        
        feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                         'MonthlyCharges', 'TotalCharges']
        
        input_data = pd.DataFrame([inputs], columns=feature_names)
        
        prediction = model.predict(input_data)[0]
        result = "Yes" if prediction == 1 else "No"
        
        return f"Customer Churn Prediction: {result}"
    
    return "Fill in the details and click 'Predict Churn'"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
