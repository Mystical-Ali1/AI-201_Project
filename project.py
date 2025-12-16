import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TransactionAnalyzer:
    """
    A class to handle loading, processing, and analyzing transaction data.
    """
    def __init__(self, file):
        self.file = file
        self.df = None
        self.required_columns = ['Amount', 'Time']

    def load_and_validate(self):
        """Loads CSV and checks if required columns exist."""
        try:
            self.df = pd.read_csv(self.file)
            
            
            missing_cols = [col for col in self.required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"The uploaded CSV is missing columns: {missing_cols}")
            
            return True
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return False

    def preprocess_data(self):
        """Cleans data and extracts time features."""
        try:
            
            self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
            
            
            self.df['Time'] = pd.to_timedelta(self.df['Time'], unit='s')
            self.df['Hour'] = self.df['Time'].dt.components.hours
            
            
            self.df.fillna(0, inplace=True)
            
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

    def detect_fraud_vectorized(self):
        """
        Uses NumPy for fast, vectorized fraud detection.
        Logic: Fraud if Amount > 2000 OR Time is between 10 PM and 6 AM.
        """
       
        high_amount = self.df['Amount'] > 2000
        odd_hours = (self.df['Hour'] < 6) | (self.df['Hour'] > 22)
        
        
        self.df['Fraud_Risk'] = np.where(high_amount | odd_hours, 1, 0)

    def get_dataframe(self):
        return self.df



st.set_page_config(page_title="Smart Transaction Scanner", layout="wide")
st.title("💳 Smart Transaction Scanner (OOP + NumPy Edition)")

uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    
    analyzer = TransactionAnalyzer(uploaded_file)

   
    if analyzer.load_and_validate():
        
      
        analyzer.preprocess_data()
        analyzer.detect_fraud_vectorized()
        
        
        df = analyzer.get_dataframe()

     
        st.subheader("📊 Overview")
        st.write(df.head())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Amount Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['Amount'], bins=50, color='skyblue', edgecolor='black')
            ax.set_xlabel("Amount")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col2:
            st.subheader("Transactions by Hour")
            fig, ax = plt.subplots()
            
            df['Hour'].value_counts().sort_index().plot(kind='bar', color='orange', ax=ax)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.subheader("🚨 Fraud Risk Summary")
        
       
        if not df.empty:
            fig, ax = plt.subplots()
            status_counts = df['Fraud_Risk'].value_counts()
            
            
            status_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
            
           
            labels = [x for x in status_counts.index]
            mapped_labels = ['Potential Fraud' if x==1 else 'Not Fraud' for x in labels]
            
            ax.set_xticklabels(mapped_labels, rotation=0)
            ax.set_ylabel("Transaction Count")
            st.pyplot(fig)

           
            st.download_button(
                label="Download Labeled CSV",
                data=df.to_csv(index=False),
                file_name="labeled_transactions.csv",
                mime="text/csv"
            )