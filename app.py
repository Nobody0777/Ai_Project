import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def main():
    st.set_page_config(page_title="Global Inflation Rate Prediction", layout="wide")
    st.title("🌍 Global Inflation Rate Prediction Tool")


    st.markdown(
        """
        <style>
        .main-header {background-color: #f0f0f5; padding: 10px; border-radius: 5px;}
        .sidebar {background-color: #f8f9fa; padding: 15px; border-radius: 5px;}
        </style>
        <div class="main-header">
        <h2>Predict Future Inflation Rates with Advanced Machine Learning</h2>
        <p>Upload your dataset, explore insights, and get predictions tailored for your selected country.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        """
        <div class="sidebar">
        <h3>Welcome!</h3>
        Please provide your details and upload the dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.subheader("User Information")
    user_name = st.sidebar.text_input("Enter your name:")
    user_email = st.sidebar.text_input("Enter your email:")

    if user_name and user_email:
        st.sidebar.success(f"Welcome, {user_name}!")

    uploaded_file = st.file_uploader("Upload your inflation dataset (CSV format):", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data = data.replace(r'^\s*$', np.nan, regex=True).fillna(0)

        st.subheader("Dataset Overview")
        st.write("Below is a preview of your dataset:")
        st.dataframe(data.head(10))

        if 'country_name' in data.columns and 'indicator_name' in data.columns:
            country = st.selectbox("Select a country for prediction:", data['country_name'].unique())
            country_data = data[data['country_name'] == country]

            try:
                country_data = country_data.drop('indicator_name', axis=1).set_index('country_name').transpose().reset_index()
                country_data.columns = ['Year', 'Inflation Rate (%)']
                country_data['Year'] = country_data['Year'].astype(int)
                country_data['Inflation Rate (%)'] = country_data['Inflation Rate (%)'].astype(float)

                st.subheader(f"Historical Data for {country}")
                st.write(country_data)

                X = country_data[['Year']]
                y = country_data['Inflation Rate (%)']

                # Scaling and polynomial features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X_scaled)

                X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

                # Ridge Regression for better accuracy
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.subheader("Model Evaluation")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-Squared (Accuracy): {r2:.2f}")

                st.markdown("### Predict Future Inflation Rates")
                start_year = st.slider("Select start year:", 2024, 2050, 2024)
                end_year = st.slider("Select end year:", 2025, 2055, 2025)

                future_years = pd.DataFrame({'Year': range(start_year, end_year + 1)})
                future_years_scaled = scaler.transform(future_years)
                future_years_poly = poly.transform(future_years_scaled)
                future_predictions = model.predict(future_years_poly)

                predictions_df = pd.DataFrame({
                    'Year': future_years['Year'],
                    'Predicted Inflation Rate (%)': future_predictions
                })

                st.subheader("Future Inflation Predictions")
                st.write(predictions_df)

                st.subheader("Visualization")
                plt.figure(figsize=(10, 6))
                plt.plot(country_data['Year'], country_data['Inflation Rate (%)'], label='Historical Data', marker='o')
                plt.plot(future_years['Year'], future_predictions, label='Predicted Data', linestyle='--')
                plt.xlabel('Year')
                plt.ylabel('Inflation Rate (%)')
                plt.title(f'Inflation Rate Prediction for {country}')
                plt.legend()
                st.pyplot(plt)

                st.download_button(
                    label="Download Prediction Data",
                    data=predictions_df.to_csv(index=False),
                    file_name=f"{country}_inflation_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error processing data for {country}: {e}")

        else:
            st.error("Required columns ('country_name', 'indicator_name') are missing.")

    else:
        st.warning("Please upload a CSV file to proceed.")

    st.sidebar.markdown(
        """
        **App Credits:**
        - Developed by Areeb, Qasim, and Hashaam
        - Powered by Streamlit
        """
    )

if __name__ == "__main__":
    main()
