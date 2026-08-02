import joblib
import pandas as pd
import streamlit as st


# ===================================================
# PAGE CONFIGURATION
# ===================================================

st.set_page_config(
    page_title="Steven's Egg Price AI",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===================================================
# CUSTOM STYLING
# ===================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #8b94a7;
            margin-bottom: 2rem;
        }

        .prediction-card {
            padding: 28px;
            border-radius: 16px;
            text-align: center;
            border: 1px solid rgba(120, 120, 120, 0.30);
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .prediction-label {
            font-size: 1rem;
            color: #8b94a7;
            margin-bottom: 6px;
        }

        .prediction-value {
            font-size: 3rem;
            font-weight: 800;
        }

        .small-note {
            font-size: 0.9rem;
            color: #8b94a7;
        }

        .footer {
            text-align: center;
            padding-top: 25px;
            padding-bottom: 20px;
            color: #8b94a7;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(120, 120, 120, 0.25);
            padding: 15px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ===================================================
# FILE PATHS AND FEATURES
# ===================================================

MODEL_PATH = "models/egg_price_model.pkl"

PREPROCESSING_PATH = (
    "models/egg_price_preprocessing.pkl"
)

EGG_HISTORY_PATH = "data/egg_prices.csv"

FEATURE_COLUMNS = [
    "corn_price",
    "soybean_price",
    "diesel_price",
    "cpi",
    "inflation_rate",
    "bird_affected",
    "bird_flu_outbreak"
]


# ===================================================
# LOAD MODEL FILES
# ===================================================

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)

    preprocessing = joblib.load(
        PREPROCESSING_PATH
    )

    return model, preprocessing


try:
    model, preprocessing = load_model_files()

except FileNotFoundError as error:
    st.error(
        "The model or preprocessing file was not found."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error("The model files could not be loaded.")
    st.code(str(error))
    st.stop()


# ===================================================
# LOAD HISTORICAL DATA
# ===================================================

@st.cache_data
def load_egg_price_history():
    history = pd.read_csv(EGG_HISTORY_PATH)

    date_candidates = [
        "date",
        "observation_date",
        "Date",
        "DATE"
    ]

    price_candidates = [
        "egg_price",
        "value",
        "Value",
        "APU0000708111"
    ]

    date_column = next(
        (
            column
            for column in date_candidates
            if column in history.columns
        ),
        None
    )

    price_column = next(
        (
            column
            for column in price_candidates
            if column in history.columns
        ),
        None
    )

    if date_column is None:
        raise ValueError(
            "No recognized date column was found."
        )

    if price_column is None:
        raise ValueError(
            "No recognized egg-price column was found."
        )

    history = history.rename(
        columns={
            date_column: "date",
            price_column: "egg_price"
        }
    )

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce"
    )

    history["egg_price"] = pd.to_numeric(
        history["egg_price"],
        errors="coerce"
    )

    history = history.dropna(
        subset=["date", "egg_price"]
    )

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    return history


try:
    egg_history = load_egg_price_history()

except Exception as error:
    egg_history = None
    history_error = str(error)


# ===================================================
# SIDEBAR
# ===================================================

with st.sidebar:

    st.title("🥚 Egg Price AI")

    st.write(
        "A machine learning application that estimates "
        "the average U.S. retail price of one dozen "
        "Grade A large eggs."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** Decision Tree Regressor")
    st.write(f"**Features:** {len(FEATURE_COLUMNS)}")
    st.write("**Target:** Monthly egg price")

    metric_column_1, metric_column_2 = st.columns(2)

    with metric_column_1:
        st.metric("MAE", "0.160")
        st.metric("R²", "0.863")

    with metric_column_2:
        st.metric("RMSE", "0.299")

    st.divider()

    st.subheader("Features Used")

    st.markdown(
        """
        - 🌽 Corn price
        - 🌱 Soybean price
        - ⛽ Diesel price
        - 🛒 Consumer Price Index
        - 📈 Inflation rate
        - 🐔 Birds affected
        - 🦠 Bird flu outbreak
        """
    )

    st.divider()

    st.subheader("Project Links")

    st.link_button(
        "View GitHub Repository",
        (
            "https://github.com/"
            "bonillasteven/egg-price-predictor"
        ),
        use_container_width=True
    )

    st.caption("Created by Steven A. Bonilla")


# ===================================================
# MAIN HEADER
# ===================================================

st.markdown(
    (
        '<div class="main-title">'
        "🥚 U.S. Egg Price Prediction System"
        "</div>"
    ),
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Estimate the monthly average retail price of one dozen
        Grade A large eggs using current economic and
        agricultural conditions.
    </div>
    """,
    unsafe_allow_html=True
)


# ===================================================
# MAIN TABS
# ===================================================

(
    prediction_tab,
    dashboard_tab,
    model_tab,
    project_tab
) = st.tabs(
    [
        "🔮 Make a Prediction",
        "📊 Data Dashboard",
        "🌳 Model Details",
        "📘 About the Project"
    ]
)


# ===================================================
# TAB 1: PREDICTION
# ===================================================

with prediction_tab:

    st.subheader("Enter Current Values")

    st.write(
        "Corn and soybean prices are entered in dollars "
        "per bushel. The app converts them to dollars per "
        "metric ton before making a prediction."
    )

    with st.form("prediction_form"):

        left_column, right_column = st.columns(2)

        with left_column:

            corn_bushel = st.number_input(
                "Corn price ($ per bushel)",
                min_value=0.0,
                max_value=30.0,
                value=4.50,
                step=0.01
            )

            soybean_bushel = st.number_input(
                "Soybean price ($ per bushel)",
                min_value=0.0,
                max_value=50.0,
                value=10.50,
                step=0.01
            )

            diesel_price = st.number_input(
                "Diesel fuel price ($ per gallon)",
                min_value=0.0,
                max_value=15.0,
                value=3.75,
                step=0.01
            )

            cpi = st.number_input(
                "Consumer Price Index (CPI)",
                min_value=0.0,
                max_value=1000.0,
                value=320.0,
                step=0.1
            )

        with right_column:

            inflation_rate = st.number_input(
                "Year-over-year inflation rate (%)",
                min_value=-20.0,
                max_value=50.0,
                value=3.0,
                step=0.1
            )

            bird_affected = st.number_input(
                "Number of birds affected by bird flu",
                min_value=0,
                max_value=500_000_000,
                value=0,
                step=1000
            )

            outbreak_choice = st.radio(
                "Was a bird flu outbreak reported?",
                options=["No", "Yes"],
                horizontal=True
            )

            show_inputs = st.checkbox(
                "Display an input summary",
                value=True
            )

        submit_button = st.form_submit_button(
            "🥚 Predict Egg Price",
            use_container_width=True
        )

    if submit_button:

        bird_flu_outbreak = (
            1 if outbreak_choice == "Yes" else 0
        )

        corn_metric_ton = corn_bushel * 39.368
        soybean_metric_ton = soybean_bushel * 36.744

        new_data = pd.DataFrame(
            {
                "corn_price": [corn_metric_ton],
                "soybean_price": [soybean_metric_ton],
                "diesel_price": [diesel_price],
                "cpi": [cpi],
                "inflation_rate": [inflation_rate],
                "bird_affected": [bird_affected],
                "bird_flu_outbreak": [
                    bird_flu_outbreak
                ]
            }
        )

        new_data = new_data[FEATURE_COLUMNS]

        try:
            new_data_prepared = (
                preprocessing.transform(new_data)
            )

            prediction = model.predict(
                new_data_prepared
            )[0]

        except Exception as error:
            st.error(
                "The prediction could not be completed."
            )
            st.code(str(error))
            st.stop()

st.markdown(
    f"""
    <div class="prediction-card">

        <p class="prediction-label">
            Estimated retail price
        </p>

        <p class="prediction-value">
            ${prediction:.2f}
        </p>

        <p class="small-note">
            Per dozen Grade A large eggs
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

    

        result_column_1, result_column_2, result_column_3 = (
            st.columns(3)
        )

        with result_column_1:
            st.metric(
                "Predicted Price",
                f"${prediction:.2f}"
            )

        with result_column_2:
            st.metric(
                "Outbreak Status",
                outbreak_choice
            )

        with result_column_3:
            st.metric(
                "Inflation",
                f"{inflation_rate:.2f}%"
            )

        if show_inputs:

            st.subheader("Input Summary")

            summary = pd.DataFrame(
                {
                    "Input": [
                        "Corn price entered",
                        "Corn price used by model",
                        "Soybean price entered",
                        "Soybean price used by model",
                        "Diesel price",
                        "Consumer Price Index",
                        "Inflation rate",
                        "Birds affected",
                        "Bird flu outbreak"
                    ],
                    "Value": [
                        f"${corn_bushel:.2f} per bushel",
                        (
                            f"${corn_metric_ton:,.2f} "
                            "per metric ton"
                        ),
                        f"${soybean_bushel:.2f} per bushel",
                        (
                            f"${soybean_metric_ton:,.2f} "
                            "per metric ton"
                        ),
                        f"${diesel_price:.2f} per gallon",
                        f"{cpi:.2f}",
                        f"{inflation_rate:.2f}%",
                        f"{bird_affected:,}",
                        outbreak_choice
                    ]
                }
            )

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )

        prediction_download = pd.DataFrame(
            {
                "predicted_egg_price": [
                    round(prediction, 2)
                ],
                "corn_price_per_bushel": [
                    corn_bushel
                ],
                "soybean_price_per_bushel": [
                    soybean_bushel
                ],
                "diesel_price": [diesel_price],
                "cpi": [cpi],
                "inflation_rate": [inflation_rate],
                "bird_affected": [bird_affected],
                "bird_flu_outbreak": [
                    bird_flu_outbreak
                ]
            }
        )

        csv_data = prediction_download.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction as CSV",
            data=csv_data,
            file_name="egg_price_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.info(
            "This result is an estimate based on historical "
            "data. It is not a guaranteed future price."
        )


        # ===========================================
        # AI INSIGHTS
        # ===========================================

        st.divider()

        st.subheader("🤖 AI Insights")

        insights = []

        if bird_flu_outbreak == 1:
            insights.append(
                "🦠 A bird flu outbreak was reported. "
                "Reduced poultry supply may place upward "
                "pressure on egg prices."
            )

        if inflation_rate >= 4:
            insights.append(
                "📈 Inflation is relatively high. "
                "Production, packaging, and transportation "
                "costs may be elevated."
            )

        elif inflation_rate < 0:
            insights.append(
                "📉 Inflation is negative, which may indicate "
                "slower overall price growth."
            )

        if diesel_price >= 4:
            insights.append(
                "⛽ Diesel prices are elevated. "
                "Higher transportation costs may increase "
                "retail egg prices."
            )

        if corn_bushel >= 6:
            insights.append(
                "🌽 Corn prices are above the typical range. "
                "Higher feed costs may increase production "
                "expenses."
            )

        if soybean_bushel >= 14:
            insights.append(
                "🌱 Soybean prices are relatively high. "
                "This may increase poultry feed costs."
            )

        if bird_affected >= 10_000_000:
            insights.append(
                "🐔 A large number of birds were affected. "
                "This may reduce egg production and tighten "
                "supply."
            )

        elif 0 < bird_affected < 10_000_000:
            insights.append(
                "🐔 Bird flu affected part of the poultry "
                "population, which may influence supply."
            )

        if prediction >= 5:
            insights.append(
                "💵 The predicted egg price is relatively "
                "high compared with many historical periods."
            )

        elif prediction <= 2:
            insights.append(
                "💵 The predicted egg price is relatively "
                "low compared with many recent periods."
            )

        if not insights:
            insights.append(
                "✅ The entered conditions appear relatively "
                "stable. No major high-cost or outbreak "
                "signals were detected."
            )

        for insight in insights:
            st.info(insight)

        st.caption(
            "These AI insights are rule-based explanations "
            "created from the entered values. They help "
            "interpret the prediction but do not prove "
            "causation."
        )
# ===================================================
# TAB 2: DATA DASHBOARD
# ===================================================

with dashboard_tab:

    st.header("📊 Egg Price Data Dashboard")

    st.write(
        "Explore historical egg prices and examine which "
        "features were most influential in the model."
    )

    # -----------------------------------------------
    # Historical egg-price chart
    # -----------------------------------------------

    st.subheader("Historical U.S. Egg Prices")

    if egg_history is not None and not egg_history.empty:

        chart_data = egg_history.set_index(
            "date"
        )[["egg_price"]]

        st.line_chart(
            chart_data,
            x_label="Date",
            y_label="Egg price ($ per dozen)",
            use_container_width=True
        )

        latest_price = egg_history.iloc[-1][
            "egg_price"
        ]

        highest_price = egg_history[
            "egg_price"
        ].max()

        average_price = egg_history[
            "egg_price"
        ].mean()

        latest_date = egg_history.iloc[-1][
            "date"
        ].strftime("%B %Y")

        price_column_1, price_column_2, price_column_3 = (
            st.columns(3)
        )

        with price_column_1:
            st.metric(
                "Latest Historical Price",
                f"${latest_price:.2f}",
                help=f"Latest observation: {latest_date}"
            )

        with price_column_2:
            st.metric(
                "Historical Average",
                f"${average_price:.2f}"
            )

        with price_column_3:
            st.metric(
                "Highest Historical Price",
                f"${highest_price:.2f}"
            )

        with st.expander(
            "View historical egg-price data"
        ):

            history_table = egg_history.copy()

            history_table["date"] = (
                history_table["date"].dt.strftime(
                    "%B %Y"
                )
            )

            history_table["egg_price"] = (
                history_table["egg_price"].round(2)
            )

            history_table = history_table.rename(
                columns={
                    "date": "Month",
                    "egg_price": "Egg Price ($)"
                }
            )

            history_table = history_table.sort_index(
                ascending=False
            )

            st.dataframe(
                history_table,
                hide_index=True,
                use_container_width=True
            )

    else:
        st.warning(
            "Historical egg-price data could not be loaded."
        )

        if "history_error" in globals():
            st.code(history_error)

        st.write(
            "Check the EGG_HISTORY_PATH value near the top "
            "of app.py and confirm the CSV filename."
        )

    st.divider()

    # -----------------------------------------------
    # Feature importance chart
    # -----------------------------------------------

    st.subheader("Decision Tree Feature Importance")

    if hasattr(model, "feature_importances_"):

        feature_importance = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": model.feature_importances_
            }
        )

        feature_importance["Importance (%)"] = (
            feature_importance["Importance"] * 100
        ).round(2)

        feature_importance = (
            feature_importance.sort_values(
                by="Importance (%)",
                ascending=True
            )
        )

        feature_chart = (
            feature_importance.set_index(
                "Feature"
            )[["Importance (%)"]]
        )

        st.bar_chart(
            feature_chart,
            horizontal=True,
            x_label="Importance (%)",
            y_label="Feature",
            use_container_width=True
        )

        display_importance = (
            feature_importance.sort_values(
                by="Importance (%)",
                ascending=False
            )
        )

        st.dataframe(
            display_importance[
                ["Feature", "Importance (%)"]
            ],
            hide_index=True,
            use_container_width=True
        )

        most_important = display_importance.iloc[0]

        st.info(
            f"The model's most influential feature was "
            f"**{most_important['Feature']}**, with an "
            f"importance score of "
            f"**{most_important['Importance (%)']:.2f}%**."
        )

        st.caption(
            "Feature importance measures predictive usefulness "
            "inside the model. It does not prove causation."
        )

    else:
        st.warning(
            "The loaded model does not provide "
            "feature importance."
        )


# ===================================================
# TAB 3: MODEL DETAILS
# ===================================================

with model_tab:

    st.header("🌳 Model Details")

    model_column_1, model_column_2, model_column_3 = (
        st.columns(3)
    )

    with model_column_1:
        st.metric(
            "Mean Absolute Error",
            "0.160"
        )

    with model_column_2:
        st.metric(
            "Root Mean Squared Error",
            "0.299"
        )

    with model_column_3:
        st.metric(
            "R² Score",
            "0.863"
        )

    st.subheader("How the Model Works")

    st.write(
        """
        The application uses a Decision Tree Regressor
        trained on historical economic and agricultural
        data. The model learns decision rules that connect
        the input variables to different egg-price levels.
        """
    )

    st.subheader("Input Features")

    feature_information = pd.DataFrame(
        {
            "Feature": [
                "Corn price",
                "Soybean price",
                "Diesel price",
                "CPI",
                "Inflation rate",
                "Birds affected",
                "Bird flu outbreak"
            ],
            "Purpose": [
                (
                    "Represents an important "
                    "poultry feed cost."
                ),
                (
                    "Represents another important "
                    "poultry feed cost."
                ),
                (
                    "Represents transportation and "
                    "distribution costs."
                ),
                (
                    "Represents the overall consumer "
                    "price level."
                ),
                (
                    "Measures year-over-year change "
                    "in CPI."
                ),
                (
                    "Represents the size of reported "
                    "bird flu events."
                ),
                (
                    "Indicates whether an outbreak "
                    "occurred."
                )
            ]
        }
    )

    st.dataframe(
        feature_information,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Model Limitations")

    st.warning(
        "The model uses historical relationships and does "
        "not include every possible cause of egg-price "
        "changes. Weather, labor costs, consumer demand, "
        "government policy, and supply-chain events may "
        "also affect actual prices."
    )


# ===================================================
# TAB 4: PROJECT INFORMATION
# ===================================================

with project_tab:

    st.header("📘 About the Project")

    st.write(
        """
        The goal of this project was to develop a machine
        learning system that predicts the monthly average
        U.S. retail price of one dozen Grade A large eggs.
        """
    )

    st.subheader("Project Workflow")

    workflow = [
        "Collect public economic and agricultural datasets",
        "Clean and merge the datasets by month",
        "Perform exploratory data analysis",
        "Create engineered features",
        "Split and preprocess the data",
        "Train several regression models",
        "Tune model hyperparameters",
        "Evaluate and compare model performance",
        "Analyze feature importance",
        "Validate chronological performance",
        "Deploy the final model with Streamlit"
    ]

    for step_number, step in enumerate(
        workflow,
        start=1
    ):
        st.write(
            f"**{step_number}.** {step}"
        )

    st.subheader("Models Compared")

    model_comparison = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Tuned Random Forest Regressor"
            ],
            "Type": [
                "Linear baseline",
                "Tree-based model",
                "Ensemble model",
                "Tuned ensemble model"
            ]
        }
    )

    st.dataframe(
        model_comparison,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Technologies Used")

    technology_columns = st.columns(4)

    with technology_columns[0]:
        st.info("🐍 Python")

    with technology_columns[1]:
        st.info("🐼 Pandas")

    with technology_columns[2]:
        st.info("🤖 Scikit-learn")

    with technology_columns[3]:
        st.info("📊 Streamlit")

    st.subheader("Future Improvements")

    st.markdown(
        """
        - Add live economic and agricultural data
        - Compare multiple prediction scenarios
        - Add XGBoost and LightGBM
        - Add SHAP prediction explanations
        - Build a true future time-series forecast
        """
    )


# ===================================================
# FOOTER
# ===================================================

st.divider()

footer_column_1, footer_column_2, footer_column_3 = (
    st.columns(3)
)

with footer_column_1:
    st.caption("Created by")
    st.write("**Steven A. Bonilla**")

with footer_column_2:
    st.caption("Machine Learning Model")
    st.write("**Decision Tree Regressor**")

with footer_column_3:
    st.caption("Built With")
    st.write(
        "**Python • Streamlit • Scikit-learn**"
    )

st.markdown(
    """
    <div class="footer">
        Steven's Egg Price AI —
        Educational and portfolio project
    </div>
    """,
    unsafe_allow_html=True
)
