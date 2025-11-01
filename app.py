import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="A/B Test Analysis",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 A/B Test Analysis for E-commerce Website")
st.markdown("""
    This application analyzes A/B test results to determine the effectiveness of website changes.
    Upload your data or use the default dataset to explore conversion rates and statistical significance.
""")

# Load data
@st.cache_data
def load_data():
    try:
        ab_data = pd.read_csv('ab_data.csv')
        countries = pd.read_csv('countries.csv')
        return ab_data, countries
    except FileNotFoundError:
        st.warning("Default data files not found. Please upload your data.")
        return None, None

ab_data, countries = load_data()

# Sidebar
st.sidebar.header("⚙️ Configuration")

if ab_data is not None:
    # Data overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(ab_data))
    with col2:
        if 'group' in ab_data.columns:
            st.metric("Control Group", len(ab_data[ab_data['group'] == 'control']))
    with col3:
        if 'group' in ab_data.columns:
            st.metric("Treatment Group", len(ab_data[ab_data['group'] == 'treatment']))
    with col4:
        if 'converted' in ab_data.columns:
            st.metric("Total Conversions", ab_data['converted'].sum())
    
    # Show raw data
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("📝 Raw Data")
        st.dataframe(ab_data.head(100))
        if countries is not None:
            st.subheader("🌍 Country Data")
            st.dataframe(countries.head(50))
    
    # Merge with countries if available
    if countries is not None and 'user_id' in ab_data.columns and 'user_id' in countries.columns:
        ab_data = ab_data.merge(countries, on='user_id', how='left')
    
    # A/B Test Analysis
    st.subheader("🔍 A/B Test Results")
    
    if 'group' in ab_data.columns and 'converted' in ab_data.columns:
        # Calculate conversion rates
        control_data = ab_data[ab_data['group'] == 'control']
        treatment_data = ab_data[ab_data['group'] == 'treatment']
        
        control_conversions = control_data['converted'].sum()
        treatment_conversions = treatment_data['converted'].sum()
        
        control_rate = control_conversions / len(control_data) * 100
        treatment_rate = treatment_conversions / len(treatment_data) * 100
        
        # Display conversion rates
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Control Conversion Rate",
                f"{control_rate:.2f}%",
                help="Percentage of users who converted in the control group"
            )
        with col2:
            st.metric(
                "Treatment Conversion Rate",
                f"{treatment_rate:.2f}%",
                delta=f"{treatment_rate - control_rate:.2f}%",
                help="Percentage of users who converted in the treatment group"
            )
        with col3:
            lift = ((treatment_rate - control_rate) / control_rate) * 100 if control_rate > 0 else 0
            st.metric(
                "Lift",
                f"{lift:.2f}%",
                help="Percentage improvement from control to treatment"
            )
        
        # Statistical significance test (Z-test for proportions)
        st.subheader("📊 Statistical Significance Test")
        
        # Perform two-proportion z-test
        control_successes = control_conversions
        control_trials = len(control_data)
        treatment_successes = treatment_conversions
        treatment_trials = len(treatment_data)
        
        # Calculate pooled probability
        pooled_prob = (control_successes + treatment_successes) / (control_trials + treatment_trials)
        pooled_se = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/control_trials + 1/treatment_trials))
        
        # Calculate z-score
        z_score = (treatment_rate/100 - control_rate/100) / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Z-Score", f"{z_score:.4f}")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
        with col3:
            significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
            st.metric("Result (α=0.05)", significance)
        
        if p_value < 0.05:
            st.success(
                f"✅ **The test is statistically significant!** "
                f"The treatment group shows a {lift:.2f}% lift with p-value = {p_value:.4f}."
            )
        else:
            st.warning(
                f"⚠️ **The test is NOT statistically significant.** "
                f"The difference could be due to random chance (p-value = {p_value:.4f})."
            )
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        # Conversion rate comparison
        fig_conv = go.Figure(data=[
            go.Bar(name='Control', x=['Control'], y=[control_rate], marker_color='lightblue'),
            go.Bar(name='Treatment', x=['Treatment'], y=[treatment_rate], marker_color='lightgreen')
        ])
        fig_conv.update_layout(
            title='Conversion Rate Comparison',
            yaxis_title='Conversion Rate (%)',
            barmode='group'
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        
        # Distribution of conversions
        conversion_data = pd.DataFrame({
            'Group': ['Control', 'Treatment'],
            'Converted': [control_conversions, treatment_conversions],
            'Not Converted': [len(control_data) - control_conversions, len(treatment_data) - treatment_conversions]
        })
        
        fig_dist = go.Figure(data=[
            go.Bar(name='Converted', x=conversion_data['Group'], y=conversion_data['Converted'], marker_color='green'),
            go.Bar(name='Not Converted', x=conversion_data['Group'], y=conversion_data['Not Converted'], marker_color='red')
        ])
        fig_dist.update_layout(
            title='Distribution of Conversions',
            yaxis_title='Number of Users',
            barmode='stack'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Time-based analysis if timestamp exists
        if 'timestamp' in ab_data.columns:
            st.subheader("📅 Time-Based Analysis")
            ab_data['timestamp'] = pd.to_datetime(ab_data['timestamp'])
            ab_data['date'] = ab_data['timestamp'].dt.date
            
            daily_conv = ab_data.groupby(['date', 'group'])['converted'].agg(['sum', 'count']).reset_index()
            daily_conv['conversion_rate'] = daily_conv['sum'] / daily_conv['count'] * 100
            
            fig_time = px.line(
                daily_conv,
                x='date',
                y='conversion_rate',
                color='group',
                title='Daily Conversion Rate Trends',
                labels={'conversion_rate': 'Conversion Rate (%)', 'date': 'Date'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Country-based analysis if available
        if 'country' in ab_data.columns:
            st.subheader("🌍 Country-Based Analysis")
            country_conv = ab_data.groupby(['country', 'group'])['converted'].agg(['sum', 'count']).reset_index()
            country_conv['conversion_rate'] = country_conv['sum'] / country_conv['count'] * 100
            
            # Top countries by traffic
            top_countries = ab_data['country'].value_counts().head(10).index
            country_conv_top = country_conv[country_conv['country'].isin(top_countries)]
            
            fig_country = px.bar(
                country_conv_top,
                x='country',
                y='conversion_rate',
                color='group',
                title='Conversion Rate by Top 10 Countries',
                labels={'conversion_rate': 'Conversion Rate (%)', 'country': 'Country'},
                barmode='group'
            )
            st.plotly_chart(fig_country, use_container_width=True)
        
        # Download results
        st.subheader("💾 Download Results")
        results_summary = pd.DataFrame({
            'Metric': ['Control Conversion Rate', 'Treatment Conversion Rate', 'Lift', 'Z-Score', 'P-Value', 'Significant (α=0.05)'],
            'Value': [f"{control_rate:.2f}%", f"{treatment_rate:.2f}%", f"{lift:.2f}%", f"{z_score:.4f}", f"{p_value:.4f}", "Yes" if p_value < 0.05 else "No"]
        })
        
        csv = results_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary Report",
            data=csv,
            file_name="ab_test_results.csv",
            mime="text/csv"
        )
    
    else:
        st.error("Required columns 'group' and 'converted' not found in the dataset.")

else:
    st.info("📂 Please upload your A/B test data to begin analysis.")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        ab_data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("### 🚀 Ready to Deploy on Streamlit Cloud")
st.markdown("""
    **Deployment Instructions:**
    1. Ensure data files (ab_data.csv, countries.csv) are in the repository
    2. Create a `requirements.txt` with: streamlit, pandas, numpy, plotly, scipy
    3. Deploy directly from GitHub via Streamlit Cloud
""")
