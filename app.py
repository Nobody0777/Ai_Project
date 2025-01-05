import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    # Custom CSS for background and fonts
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #ffffff 20%, #e6e6e6 80%);
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #333;
            color: white;
        }
        h1, h2, h3 {
            color: #004080;
        }
        .stButton>button {
            background-color: #004080;
            color: white;
            border-radius: 5px;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main header
    st.title("🌍 Global Inflation Rate Prediction Tool")
    st.markdown(
        """
        Welcome to the **Global Inflation Rate Prediction Tool**! 
        This app allows you to upload historical inflation data, explore trends, and predict future rates.
        """
    )

    # Sidebar user information
    with st.sidebar:
        st.subheader("🧍 User Information")
        user_name = st.text_input("Enter your name:")
        user_age = st.number_input("Enter your age:", min_value=1, step=1)
        user_email = st.text_input("Enter your email:")

        if user_name and user_email:
            st.success(f"Welcome, {user_name}!")

        st.markdown("---")
        st.subheader("📚 About the Tool")
        st.markdown(
            """
            This tool uses **machine learning** to analyze inflation data and make future predictions. 
            Powered by **Linear Regression**.
            """
        )
        st.markdown("---")

    # File upload
    st.markdown("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file containing inflation data:", type="csv")

    if uploaded_file:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)

            # Data cleaning
            data.columns = [col.strip() for col in data.columns]  # Remove whitespace from column names
            data = data.replace(r"^\s*$", np.nan, regex=True)  # Replace empty strings with NaN
            data.dropna(inplace=True)  # Drop rows with NaN values

            # Ensure proper data types
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Inflation Rate'] = pd.to_numeric(data['Inflation Rate'], errors='coerce')
            data.dropna(inplace=True)

            st.markdown("### 📊 Data Overview")
            st.write(data.head(10))

            # Ensure columns exist for proper processing
            if 'Year' in data.columns and 'Inflation Rate' in data.columns:
                st.markdown("### 📈 Data Visualization")
                fig, ax = plt.subplots()
                sns.lineplot(data=data, x='Year', y='Inflation Rate', marker="o", ax=ax)
                ax.set_title("Inflation Rate Over Time")
                ax.set_xlabel("Year")
                ax.set_ylabel("Inflation Rate (%)")
                st.pyplot(fig)

                # Predictive Analysis
                st.markdown("### 🤖 Predictive Analysis")
                X = data[['Year']]
                y = data['Inflation Rate']

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Linear Regression Model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Display predictions vs actual
                result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write(result_df)

                # R^2 score
                st.write(f"Model Performance (R^2 Score): {model.score(X_test, y_test):.2f}")

                # Future Predictions
                st.markdown("### 🔮 Future Predictions")
                future_years = st.number_input("Enter the number of future years to predict:", min_value=1, step=1)
                if future_years:
                    last_year = data['Year'].max()
                    future_X = pd.DataFrame({"Year": [last_year + i for i in range(1, future_years + 1)]})
                    future_predictions = model.predict(future_X)

                    future_df = pd.DataFrame({"Year": future_X['Year'], "Predicted Inflation Rate": future_predictions})
                    st.write(future_df)

                    # Plot future predictions
                    fig, ax = plt.subplots()
                    sns.lineplot(data=future_df, x='Year', y='Predicted Inflation Rate', marker="o", ax=ax, color="orange")
                    ax.set_title("Future Inflation Rate Predictions")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Inflation Rate (%)")
                    st.pyplot(fig)

            else:
                st.error("The uploaded file must contain 'Year' and 'Inflation Rate' columns.")

        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Please upload a CSV file to begin.")

    st.markdown("---")
    st.markdown("### 🤝 Collaborators")
    st.sidebar.markdown(
        """
        **Contributors**:
        - Muhammad Areeb
        - Qasim Tahir
        - Hashaam Amjad
        """
    )

if __name__ == "__main__":
    main()

