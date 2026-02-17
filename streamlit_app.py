# streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Instacart Recommender",
    page_icon="🛒",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Title
st.title("🛒 Instacart Recommendation System")
st.markdown("*Personalized grocery recommendations powered by LightGBM*")

# Sidebar
st.sidebar.header("Configuration")
user_id = st.sidebar.number_input(
    "User ID", 
    min_value=1, 
    max_value=206209, 
    value=1,
    help="Enter a user ID (1-206,209)"
)
k = st.sidebar.slider("Number of Recommendations", 1, 20, 10)
strategy = st.sidebar.selectbox(
    "Strategy",
    ["auto", "model", "popularity"],
    help="auto: Choose based on user history | model: Force ML model | popularity: Popular items"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Recommendations")
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={
                        "user_id": int(user_id),
                        "k": k,
                        "strategy": strategy
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display results
                    st.success(f"✅ Generated {len(data['recommendations'])} recommendations in {data['latency_ms']:.1f}ms")
                    
                    # User info badge
                    user_type_color = {
                        "warm": "🟢",
                        "cold": "🟡", 
                        "new": "🔴"
                    }
                    st.info(f"{user_type_color.get(data['user_type'], '⚪')} User Type: **{data['user_type'].upper()}** | Strategy: **{data['strategy_used']}**")
                    
                    # Recommendations table
                    recs_df = pd.DataFrame(data['recommendations'])
                    
                    # Format as a nice table
                    st.subheader("Top Products")
                    for idx, rec in enumerate(data['recommendations'], 1):
                        col_a, col_b, col_c = st.columns([1, 4, 2])
                        with col_a:
                            st.metric("Rank", f"#{idx}")
                        with col_b:
                            st.markdown(f"**{rec['product_name']}**")
                            st.caption(rec.get('reason', 'Recommended for you'))
                        with col_c:
                            st.metric("Score", f"{rec['score']:.3f}")
                        
                        st.divider()
                    
                    # Chart
                    st.subheader("Score Distribution")
                    fig = px.bar(
                        recs_df,
                        x='product_name',
                        y='score',
                        labels={'product_name': 'Product', 'score': 'Confidence Score'},
                        color='score',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the API is running on http://localhost:8000")
                st.info("Start the API with: `uvicorn src.api.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.header("📊 System Info")
    
    # Health check
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API is healthy")
            st.metric("Model", health_data['model_version'])
            st.metric("Features", health_data['features_count'])
        else:
            st.error("❌ API unhealthy")
    except:
        st.error("❌ API not running")
    
    # About
    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    This recommendation system predicts which grocery products 
    a user will reorder based on their purchase history.
    
    **Features:**
    - 45 engineered features
    - LightGBM model (F1: 0.507)
    - Cold start handling
    - Per-user F1 optimization
    
    **User Types:**
    - 🟢 Warm: 5+ orders
    - 🟡 Cold: 1-4 orders  
    - 🔴 New: 0 orders
    """)
    
    # Sample user IDs
    st.divider()
    st.subheader("🎲 Try These Users")
    st.markdown("""
    - User 1: Frequent shopper
    - User 100: Regular user
    - User 1000: Average user
    - User 200000: Edge case
    """)

# Footer
st.divider()
st.caption("Built with FastAPI + LightGBM + Streamlit | [GitHub](https://github.com/BessiePengjinWang/instacart-recommendation-system)")