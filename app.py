import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="Steven's Egg Price AI",
    page_icon="🥚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🥚 Egg Price AI")

    st.write(
        "A machine learning application that estimates "
        "the average U.S. retail price of one dozen eggs."
    )

    st.markdown("---")

    st.subheader("Model Inputs")

    st.write(
        "• Corn price\n"
        "• Soybean price\n"
        "• Diesel price\n"
        "• CPI\n"
        "• Inflation rate\n"
        "• Birds affected\n"
        "• Bird flu outbreak"
    )

    st.markdown("---")

    st.caption(
        "Created by Steven A. Bonilla"
    )

# Main heading
st.title("🥚 U.S. Egg Price Prediction System")

st.markdown(
    """
    Use the form below to estimate the average retail price of
    **one dozen Grade A large eggs** based on current economic
    and agricultural conditions.
    """
)


# ---------------------------------------------------
# Load the trained model and preprocessing pipeline
# ---------------------------------------------------

@st.cache_resource
def load_model_files():
    model = joblib.load(
        "models/egg_price_model.pkl"
    )

    preprocessing = joblib.load(
        "models/egg_price_preprocessing.pkl"
    )

    return model, preprocessing


try:
    model, preprocessing = load_model_files()

except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()


# ---------------------------------------------------
# Current model features
# ---------------------------------------------------

FEATURE_COLUMNS = [
    "corn_price",
    "soybean_price",
    "diesel_price",
    "cpi",
    "inflation_rate",
    "bird_affected",
    "bird_flu_outbreak"
]


# ---------------------------------------------------
# User input form
# ---------------------------------------------------

with st.form("prediction_form"):

    left_column, right_column = st.columns(2)

    with left_column:

        corn_bushel = st.number_input(
            "Corn price ($ per bushel)",
            min_value=0.0,
            value=4.50,
            step=0.01,
            help=(
                "Enter the corn price in dollars per bushel. "
                "The application converts it to dollars per metric ton."
            )
        )

        soybean_bushel = st.number_input(
            "Soybean price ($ per bushel)",
            min_value=0.0,
            value=10.50,
            step=0.01,
            help=(
                "Enter the soybean price in dollars per bushel. "
                "The application converts it to dollars per metric ton."
            )
        )

        diesel_price = st.number_input(
            "Diesel fuel price ($ per gallon)",
            min_value=0.0,
            value=3.75,
            step=0.01,
            help="Enter the average diesel price per gallon."
        )

        cpi = st.number_input(
            "Consumer Price Index (CPI)",
            min_value=0.0,
            value=320.0,
            step=0.1,
            help="Enter the CPI index value."
        )

    with right_column:

        inflation_rate = st.number_input(
            "Year-over-year inflation rate (%)",
            value=3.0,
            step=0.1,
            help="For example, enter 3.2 for an inflation rate of 3.2%."
        )

        bird_affected = st.number_input(
            "Number of birds affected by bird flu",
            min_value=0,
            value=0,
            step=1000,
            help="Enter the total number of birds affected."
        )

        outbreak_choice = st.radio(
            "Was a bird flu outbreak reported?",
            options=["No", "Yes"],
            horizontal=True
        )

        show_inputs = st.checkbox(
            "Show entered values",
            value=True
        )

    submit_button = st.form_submit_button(
        "Predict Egg Price",
        use_container_width=True
    )


# ---------------------------------------------------
# Make prediction
# ---------------------------------------------------

if submit_button:

    # Convert Yes/No into the binary feature used during training
    bird_flu_outbreak = (
        1 if outbreak_choice == "Yes" else 0
    )

    # Convert dollars per bushel into dollars per metric ton
    corn_metric_ton = corn_bushel * 39.368
    soybean_metric_ton = soybean_bushel * 36.744

    # Create input using the exact feature names used during training
    new_data = pd.DataFrame({
        "corn_price": [corn_metric_ton],
        "soybean_price": [soybean_metric_ton],
        "diesel_price": [diesel_price],
        "cpi": [cpi],
        "inflation_rate": [inflation_rate],
        "bird_affected": [bird_affected],
        "bird_flu_outbreak": [bird_flu_outbreak]
    })

    # Preserve the same column order used during model training
    new_data = new_data[FEATURE_COLUMNS]

    try:
        # Apply the preprocessing fitted on the training data
        new_data_prepared = preprocessing.transform(
            new_data
        )

        # Use the saved Decision Tree model
        prediction = model.predict(
            new_data_prepared
        )[0]

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.success(
        f"Predicted egg price: "
        f"${prediction:.2f} per dozen"
    )

    st.metric(
        label="Estimated Egg Price",
        value=f"${prediction:.2f} per dozen"
    )

    if show_inputs:

        st.subheader("Input Summary")

        summary = pd.DataFrame({
            "Input": [
                "Corn price",
                "Corn price used by model",
                "Soybean price",
                "Soybean price used by model",
                "Diesel price",
                "CPI",
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
        })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

    st.info(
        "This prediction is an estimate based on historical "
        "economic and agricultural data. It is not a guaranteed "
        "future retail price."
    )


# ---------------------------------------------------
# Project information
# ---------------------------------------------------

st.divider()

st.subheader("About the Model")

st.write(
    "This application uses a Decision Tree regression model trained "
    "on historical egg prices, feed prices, diesel prices, CPI, "
    "inflation, and bird flu outbreak information."
)

st.caption(
    "Created by Steven A. Bonilla as a machine learning project."
)
