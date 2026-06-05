import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="AI-Powered E-Commerce Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_excel(
        "Online Retail.xlsx",
        engine="openpyxl"
    )

    df.drop_duplicates(inplace=True)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].fillna("Unknown")

    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    if "InvoiceDate" in df.columns:
        df["Month"] = df["InvoiceDate"].dt.month

    return df

df = load_data()

st.sidebar.title("Filters")

if "Country" in df.columns:

    countries = ["All"] + sorted(
        df["Country"].dropna().astype(str).unique().tolist()
    )

    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries
    )

    if selected_country != "All":
        df = df[df["Country"].astype(str) == selected_country]

if df.empty:
    st.warning("No data available for selected country.")
    st.stop()

st.title("AI-Powered E-Commerce Customer Analytics Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", len(df))

with col2:
    if "Revenue" in df.columns:
        st.metric(
            "Total Revenue",
            f"${df['Revenue'].sum():,.2f}"
        )

with col3:
    if "CustomerID" in df.columns:
        st.metric(
            "Customers",
            df["CustomerID"].nunique()
        )

with col4:
    if "Country" in df.columns:
        st.metric(
            "Countries",
            df["Country"].nunique()
        )

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.header("Sales Analytics")

col1, col2 = st.columns(2)

with col1:

    if "Country" in df.columns and "Revenue" in df.columns:

        st.subheader("Revenue by Country")

        country_sales = (
            df.groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        if not country_sales.empty:

            fig, ax = plt.subplots(figsize=(8, 5))

            country_sales.plot(
                kind="bar",
                ax=ax
            )

            plt.xticks(rotation=45)

            st.pyplot(fig)

with col2:

    if "Month" in df.columns and "Revenue" in df.columns:

        st.subheader("Monthly Revenue Trend")

        monthly_sales = (
            df.groupby("Month")["Revenue"]
            .sum()
            .sort_index()
        )

        if not monthly_sales.empty:

            fig, ax = plt.subplots(figsize=(8, 5))

            monthly_sales.plot(
                kind="line",
                marker="o",
                ax=ax
            )

            st.pyplot(fig)

st.header("Data Quality Analysis")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Missing Values")

    missing = df.isnull().sum()
    missing_values = missing[missing > 0]

    if missing_values.empty:

        st.success("No missing values found.")

    else:

        fig, ax = plt.subplots(figsize=(8, 4))

        missing_values.plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

with col2:

    if "Revenue" in df.columns:

        st.subheader("Revenue Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(
            df["Revenue"],
            bins=30,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

if "Revenue" in df.columns:

    st.header("Outlier Detection")

    fig, ax = plt.subplots(figsize=(10, 2))

    sns.boxplot(
        x=df["Revenue"],
        ax=ax
    )

    st.pyplot(fig)

st.header("Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

if len(df) > 1 and len(numeric_df.columns) > 1:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

if "Revenue" in df.columns:

    st.header("Revenue Log Transformation")

    revenue_log = np.log1p(
        df["Revenue"].clip(lower=0)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Revenue")

        fig, ax = plt.subplots()

        sns.histplot(
            df["Revenue"],
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Log Transformed")

        fig, ax = plt.subplots()

        sns.histplot(
            revenue_log,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

st.header("Full Dataset")

st.dataframe(df)

csv = df.to_csv(index=False)

st.download_button(
    label="Download Processed Dataset",
    data=csv,
    file_name="processed_retail_data.csv",
    mime="text/csv"
)