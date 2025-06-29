import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="GNN Cyber Defense System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for futuristic theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

.main {
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    color: #00ff88;
}

.stApp {
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
}

.title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(45deg, #00ff88, #00d4aa, #0099cc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
    margin-bottom: 2rem;
}

.metric-card {
    background: rgba(0, 255, 136, 0.1);
    border: 2px solid #00ff88;
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    backdrop-filter: blur(10px);
}

.alert-high {
    background: rgba(255, 0, 100, 0.2);
    border: 2px solid #ff0064;
    border-radius: 10px;
    padding: 15px;
    margin: 5px 0;
    box-shadow: 0 0 15px rgba(255, 0, 100, 0.4);
    animation: pulse 2s infinite;
}

.alert-normal {
    background: rgba(0, 255, 136, 0.1);
    border: 2px solid #00ff88;
    border-radius: 10px;
    padding: 15px;
    margin: 5px 0;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
}

@keyframes pulse {
    0% { box-shadow: 0 0 15px rgba(255, 0, 100, 0.4); }
    50% { box-shadow: 0 0 25px rgba(255, 0, 100, 0.7); }
    100% { box-shadow: 0 0 15px rgba(255, 0, 100, 0.4); }
}

.node-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.node-card {
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid #00ff88;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    transition: all 0.3s ease;
}

.node-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
}

.node-attack {
    background: rgba(255, 0, 100, 0.2);
    border-color: #ff0064;
    animation: pulse 2s infinite;
}

.sidebar .sidebar-content {
    background: rgba(0, 0, 0, 0.8);
    border-right: 2px solid #00ff88;
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace;
    color: #00ff88;
}

.stMetric {
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid #00ff88;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

def load_predictions():
    """Load the latest predictions from the JSON file"""
    try:
        with open("latest_predictions.json", "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return dummy data if file doesn't exist
        return pd.DataFrame({
            "timestamp": [datetime.now().isoformat()],
            "node_id": ["node_001"],
            "prediction": ["NORMAL"],
            "confidence": [0.85]
        })

def create_attack_distribution_chart(df):
    """Create attack distribution pie chart"""
    attack_counts = df['prediction'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=attack_counts.index,
        values=attack_counts.values,
        hole=0.4,
        marker=dict(
            colors=['#00ff88', '#ff0064'],
            line=dict(color='#000000', width=2)
        ),
        textfont=dict(size=16, color='white')
    )])
    
    fig.update_layout(
        title={
            'text': '🛡️ THREAT ANALYSIS',
            'x': 0.5,
            'font': {'size': 20, 'color': '#00ff88', 'family': 'Orbitron'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ff88', family='Orbitron'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#00ff88')
        )
    )
    
    return fig

def create_confidence_gauge(avg_confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = avg_confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "SYSTEM CONFIDENCE", 'font': {'size': 20, 'color': '#00ff88', 'family': 'Orbitron'}},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#00ff88'},
            'bar': {'color': "#00ff88"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 0, 100, 0.3)"},
                {'range': [50, 80], 'color': "rgba(255, 255, 0, 0.3)"},
                {'range': [80, 100], 'color': "rgba(0, 255, 136, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#00ff88', 'family': 'Orbitron'},
        height=300
    )
    
    return fig

def create_timeline_chart(df):
    """Create high-resolution microsecond-level timeline of predictions"""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp')
    
    attack_data = df[df['prediction'] == 'ATTACK']
    normal_data = df[df['prediction'] == 'NORMAL']
    
    fig = go.Figure()

    # NORMAL
    fig.add_trace(go.Scatter(
        x=normal_data['timestamp'],
        y=[1] * len(normal_data),
        mode='markers',
        marker=dict(size=10, color='#00ff88', symbol='circle'),
        name='NORMAL',
        hovertemplate='<b>NORMAL</b><br>Time: %{x|%H:%M:%S.%f}<br>Confidence: %{customdata:.2f}<extra></extra>',
        customdata=normal_data['confidence']
    ))

    # ATTACK
    fig.add_trace(go.Scatter(
        x=attack_data['timestamp'],
        y=[2] * len(attack_data),
        mode='markers',
        marker=dict(size=15, color='#ff0064', symbol='triangle-up'),
        name='ATTACK',
        hovertemplate='<b>⚠️ ATTACK DETECTED</b><br>Time: %{x|%H:%M:%S.%f}<br>Confidence: %{customdata:.2f}<extra></extra>',
        customdata=attack_data['confidence']
    ))

    fig.update_layout(
        title={
            'text': '⏱️ REAL-TIME THREAT TIMELINE',
            'x': 0.5,
            'font': {'size': 20, 'color': '#00ff88', 'family': 'Orbitron'}
        },
        xaxis=dict(
            title='Time (HH:MM:SS.microseconds)',
            tickformat='%H:%M:%S.%f',
            showgrid=True
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2],
            ticktext=['NORMAL', 'ATTACK'],
            range=[0.5, 2.5]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#00ff88', 'family': 'Orbitron'},
        showlegend=True,
        legend=dict(font=dict(color='#00ff88')),
        height=350
    )

    return fig

def display_node_status(df):
    """Display current status of all nodes"""
    st.markdown("### 🌐 NODE STATUS MATRIX")
    
    # Get latest status for each node
    latest_status = df.groupby('node_id').first().reset_index()
    
    # Create columns for grid layout
    cols = st.columns(min(4, len(latest_status)))
    
    for idx, (_, node) in enumerate(latest_status.iterrows()):
        col_idx = idx % len(cols)
        with cols[col_idx]:
            status = node['prediction']
            confidence = node['confidence']
            node_id = node['node_id']
            
            if status == 'ATTACK':
                st.markdown(f"""
                <div class="node-card node-attack">
                    <h4>🚨 {node_id}</h4>
                    <p><strong>ATTACK DETECTED</strong></p>
                    <p>Confidence: {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="node-card">
                    <h4>✅ {node_id}</h4>
                    <p><strong>SECURE</strong></p>
                    <p>Confidence: {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

from streamlit_autorefresh import st_autorefresh

def main():
    # Title
    st.markdown('<h1 class="title">🛡️ GNN CYBER DEFENSE SYSTEM</h1>', unsafe_allow_html=True)

    # Checkbox to enable/disable auto-refresh
    auto_refresh = st.checkbox("🔄 Auto Refresh (5s)", value=True)

    if auto_refresh:
        # This will auto-refresh the whole page every 5 seconds
        st_autorefresh(interval=5000, key="datarefresh")

    # Load data
    df_full = load_predictions()
    if not df_full.empty:
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        df = df_full.sort_values('timestamp', ascending=False).head(25)  # Latest 10 only
    else:
        df = df_full

    if df.empty:
        st.warning("⚠️ No prediction data available. Make sure your inference script is running.")
        return

    # Dashboard content...
    total_predictions = len(df)
    attack_count = len(df[df['prediction'] == 'ATTACK'])
    attack_rate = (attack_count / total_predictions) * 100 if total_predictions > 0 else 0
    avg_confidence = df['confidence'].mean()
    unique_nodes = df['node_id'].nunique()

    # ... add charts, metrics, visuals here ...

        
        # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 LATEST SCANS",
            value=total_predictions,
            delta="Last 10"
        )
    
    with col2:
        st.metric(
            label="🚨 ATTACKS DETECTED",
            value=attack_count,
            delta=f"{attack_rate:.1f}% rate",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🎲 AVG CONFIDENCE",
            value=f"{avg_confidence:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🌐 ACTIVE NODES",
            value=unique_nodes,
            delta="Current batch"
        )
    
    with col5:
        threat_level = "🔴 HIGH" if attack_rate > 20 else "🟡 MEDIUM" if attack_rate > 5 else "🟢 LOW"
        st.metric(
            label="⚡ THREAT LEVEL",
            value=threat_level,
            delta=None
        )
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Attack distribution
        fig_pie = create_attack_distribution_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence gauge
        fig_gauge = create_confidence_gauge(avg_confidence)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Timeline chart
    fig_timeline = create_timeline_chart(df)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Node status grid
    display_node_status(df)
    
    st.markdown("---")
    
    # Recent alerts (showing all 10 since we already limited the dataset)
    st.markdown("### 🚨 RECENT ALERTS")
    
    # Sort by timestamp, most recent first
    df_sorted = df.sort_values('timestamp', ascending=False)
    
    for _, row in df_sorted.iterrows():
        timestamp = pd.to_datetime(row['timestamp']).strftime('%H:%M:%S.%f')
        node_id = row['node_id']
        prediction = row['prediction']
        confidence = row['confidence']
        
        if prediction == 'ATTACK':
            st.markdown(f"""
            <div class="alert-high">
                <strong>🚨 ATTACK DETECTED</strong> | 
                Node: <code>{node_id}</code> | 
                Time: <code>{timestamp}</code> | 
                Confidence: <strong>{confidence:.2f}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-normal">
                <strong>✅ NORMAL TRAFFIC</strong> | 
                Node: <code>{node_id}</code> | 
                Time: <code>{timestamp}</code> | 
                Confidence: <strong>{confidence:.2f}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #00ff88; font-family: Orbitron;">🛡️ GNN-Based Network Intrusion Detection System | Real-time Cyber Defense</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()