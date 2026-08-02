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
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .prediction-card {
            padding: 28px;
            border-radius: 16px;
            text-align: center;
            border: 1px solid rgba(120, 120, 120, 0.25);
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .prediction-label {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 5px;
        }

        .prediction-value {
            font-size: 3rem;
            font-weight: 800;
        }

        .small-note {
            font-size: 0.9rem;
            color: #6b7280;
        }

        .footer {
            text-align: center;
            padding-top: 25px;
            padding-bottom: 20px;
            color: #6b7280;
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
# FILE PATHS AND MODEL FEATURES
# ===================================================

MODEL_PATH = "models/egg_price_model.pkl"
PREPROCESSING_PATH = "models/egg_price_preprocessing.pkl"

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
    """
    Load the trained model and preprocessing pipeline.
    """

    trained_model = joblib.load(MODEL_PATH)
    trained_preprocessing = joblib.load(PREPROCESSING_PATH)

    return trained_model, trained_preprocessing


try:
    model, preprocessing = load_model_files()

except FileNotFoundError as error:
    st.error(
        "The model or preprocessing file could not be found. "
        "Confirm that both files are inside the models folder."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error("The model files could not be loaded.")
    st.code(str(error))
    st.stop()


# ===================================================
# SIDEBAR
# ===================================================

with st.sidebar:

    st.title("🥚 Egg Price AI")

    st.write(
        "A machine learning application that estimates the "
        "average U.S. retail price of one dozen Grade A large eggs."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** Decision Tree Regressor")
    st.write(f"**Number of features:** {len(FEATURE_COLUMNS)}")
    st.write("**Target:** Monthly egg price")

    # Update these numbers if retraining changes your scores.
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
        "www.linkedin.com/in/steven-bonilla-a26185400",
        "https://github.com/bonillasteven/egg-price-predictor",
        use_container_width=True
    )

    st.caption("Created by Steven A. Bonilla")


# ===================================================
# MAIN HEADER
# ===================================================

st.markdown(
    '<div class="main-title">🥚 U.S. Egg Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Estimate the monthly average retail price of one dozen Grade A
        large eggs using current economic and agricultural conditions.
    </div>
    """,
    unsafe_allow_html=True
)


# ===================================================
# MAIN TABS
# ===================================================

prediction_tab, model_tab, project_tab = st.tabs(
    [
        "🔮 Make a Prediction",
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
        "Corn and soybean prices can be entered in dollars per bushel. "
        "The application automatically converts them into dollars per "
        "metric ton before sending them to the model."
    )

    with st.form("prediction_form"):

        left_column, right_column = st.columns(2)

        with left_column:

            corn_bushel = st.number_input(
                "Corn price ($ per bushel)",
                min_value=0.0,
                max_value=30.0,
                value=4.50,
                step=0.01,
                help="Typical historical range is approximately $3 to $8."
            )

            soybean_bushel = st.number_input(
                "Soybean price ($ per bushel)",
                min_value=0.0,
                max_value=50.0,
                value=10.50,
                step=0.01,
                help="Typical historical range is approximately $8 to $18."
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
                step=0.1,
                help="Enter 3.2 to represent 3.2% inflation."
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


    # ===============================================
    # RUN PREDICTION
    # ===============================================

    if submit_button:

        bird_flu_outbreak = (
            1 if outbreak_choice == "Yes" else 0
        )

        # Convert dollars per bushel to dollars per metric ton.
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
                "bird_flu_outbreak": [bird_flu_outbreak]
            }
        )

        # Ensure the exact feature order used during training.
        new_data = new_data[FEATURE_COLUMNS]

        try:
            new_data_prepared = preprocessing.transform(new_data)

            prediction = model.predict(
                new_data_prepared
            )[0]

        except Exception as error:
            st.error("The prediction could not be completed.")
            st.code(str(error))
            st.stop()

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Estimated retail price
                </div>

                <div class="prediction-value">
                    ${prediction:.2f}
                </div>

                <div class="small-note">
                    Per dozen Grade A large eggs
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        result_column_1, result_column_2, result_column_3 = st.columns(3)

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
                        f"${corn_metric_ton:,.2f} per metric ton",
                        f"${soybean_bushel:.2f} per bushel",
                        f"${soybean_metric_ton:,.2f} per metric ton",
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
                "predicted_egg_price": [round(prediction, 2)],
                "corn_price_per_bushel": [corn_bushel],
                "soybean_price_per_bushel": [soybean_bushel],
                "diesel_price": [diesel_price],
                "cpi": [cpi],
                "inflation_rate": [inflation_rate],
                "bird_affected": [bird_affected],
                "bird_flu_outbreak": [bird_flu_outbreak]
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
            "This result is an estimate based on historical data. "
            "It should not be considered a guaranteed future retail price."
        )


# ===================================================
# TAB 2: MODEL DETAILS
# ===================================================

with model_tab:

    st.header("Model Details")

    model_column_1, model_column_2, model_column_3 = st.columns(3)

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
        The application uses a Decision Tree Regressor trained on
        historical economic and agricultural data. The model divides
        observations into decision-based groups and learns patterns
        associated with different egg-price levels.
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
                "Represents an important poultry feed cost.",
                "Represents another important poultry feed cost.",
                "Represents transportation and distribution costs.",
                "Represents the overall consumer price level.",
                "Measures year-over-year change in CPI.",
                "Represents the size of reported bird flu events.",
                "Indicates whether an outbreak occurred."
            ]
        }
    )

    st.dataframe(
        feature_information,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):

        feature_importance = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": model.feature_importances_
            }
        )

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

        feature_importance = feature_importance.set_index(
            "Feature"
        )

        st.bar_chart(
            feature_importance
        )

        st.caption(
            "Higher values indicate that the feature was more useful "
            "to the model. Feature importance does not prove causation."
        )

    else:
        st.info(
            "The loaded model does not provide tree-based "
            "feature importance."
        )


# ===================================================
# TAB 3: PROJECT INFORMATION
# ===================================================

with project_tab:

    st.header("About the Project")

    st.write(
        """
        The goal of this project was to develop a machine learning
        system that predicts the monthly average U.S. retail price of
        one dozen Grade A large eggs.
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
        "Deploy the final model with Streamlit"
    ]

    for step_number, step in enumerate(
        workflow,
        start=1
    ):
        st.write(f"**{step_number}.** {step}")

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

    st.subheader("Important Limitation")

    st.warning(
        "The model is trained on historical data and does not include "
        "every possible cause of egg-price changes. Weather, labor costs, "
        "consumer demand, government policies, and unexpected supply-chain "
        "events may also influence actual retail prices."
    )


# ===================================================
# FOOTER
# ===================================================

st.divider()

footer_column_1, footer_column_2, footer_column_3 = st.columns(3)

with footer_column_1:
    st.caption("Created by")
    st.write("**Steven A. Bonilla**")

with footer_column_2:
    st.caption("Machine Learning Model")
    st.write("**Decision Tree Regressor**")

with footer_column_3:
    st.caption("Built With")
    st.write("**Python • Streamlit • Scikit-learn**")

st.markdown(
    """
    <div class="footer">
        Steven's Egg Price AI — Educational and portfolio project
    </div>
    """,
    unsafe_allow_html=True
)
