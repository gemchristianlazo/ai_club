import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(
    page_title="AI Club: Fuzzy Grader", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions (The Math Engine) ---

def triangular_membership(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trapezoidal_left(x, a, b):
    return np.clip((b - x) / (b - a + 1e-9), 0, 1)

def trapezoidal_right(x, a, b):
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)

def get_university_grade(score):
    s = round(score)
    if s >= 98: return "1.00", "Excellent", "success"
    elif s >= 95: return "1.25", "Very Superior", "success"
    elif s >= 92: return "1.50", "Superior", "success"
    elif s >= 89: return "1.75", "High Average", "success"
    elif s >= 86: return "2.00", "Average", "warning"
    elif s >= 83: return "2.25", "Low Average", "warning"
    elif s >= 80: return "2.50", "Satisfactory", "warning"
    elif s >= 77: return "2.75", "Fair", "warning"
    elif s >= 75: return "3.00", "Passed", "warning"
    else: return "5.00", "Failed", "error"

# --- MAPPING FUNCTION (5-Point Scale) ---
def map_input_to_value(text_input, category):
    text_input = str(text_input).strip().lower()
    
    # Accuracy Mapping (0-100)
    if category == 'acc':
        if 'excellent' in text_input: return 98.0
        if 'good' in text_input: return 90.0
        if 'average' in text_input or 'mid' in text_input: return 82.0 
        if 'poor' in text_input: return 70.0
        return 50.0 # Very Bad
        
    # Writing/Time Mapping (0-10)
    else: 
        if 'excellent' in text_input: return 9.5
        if 'good' in text_input: return 8.0
        if 'average' in text_input or 'mid' in text_input: return 6.0
        if 'poor' in text_input: return 4.0
        return 2.0 # Very Bad

# --- Fuzzy Logic System ---

class GradingSystem:
    def __init__(self):
        self.x_acc = np.arange(0, 101, 1)
        self.x_wri = np.arange(0, 11, 0.1)
        self.x_time = np.arange(0, 11, 0.1)
        self.x_out = np.arange(0, 116, 1)

    def fuzzify(self, accuracy, writing, timeliness, params):
        p_acc_high = params.get('acc_high_start', 65)
        p_wri_good = params.get('wri_good_start', 5.0)
        p_time_good = params.get('time_good_start', 5.0)

        # Accuracy Logic
        acc_med_peak = p_acc_high
        acc_low_end = p_acc_high - 5 
        
        acc_low = trapezoidal_left(accuracy, 40, acc_low_end)
        acc_med = triangular_membership(accuracy, 40, acc_med_peak, 90)
        acc_high = trapezoidal_right(accuracy, p_acc_high, 90)

        # Writing Logic
        wri_peak = p_wri_good
        wri_low_end = p_wri_good

        wri_poor = trapezoidal_left(writing, 2, wri_low_end)
        wri_avg = triangular_membership(writing, 2, wri_peak, 8)
        wri_good = trapezoidal_right(writing, p_wri_good, 8)

        # Timeliness Logic
        time_peak = p_time_good
        time_low_end = p_time_good

        time_late = trapezoidal_left(timeliness, 2, time_low_end)
        time_ok = triangular_membership(timeliness, 2, time_peak, 8)
        time_early = trapezoidal_right(timeliness, p_time_good, 8)

        return {
            'acc': (acc_low, acc_med, acc_high),
            'wri': (wri_poor, wri_avg, wri_good),
            'time': (time_late, time_ok, time_early)
        }

    def evaluate_rules(self, f):
        acc_low, acc_med, acc_high = f['acc']
        wri_poor, wri_avg, wri_good = f['wri']
        time_late, time_ok, time_early = f['time']

        # Rule Definitions
        fail_cond = np.fmax(acc_low, np.fmin(acc_med, np.fmax(wri_poor, time_late)))
        
        habits_acceptable = np.fmax(wri_avg, np.fmax(time_ok, np.fmax(wri_good, time_early)))
        pass_cond = np.fmax(np.fmin(acc_med, habits_acceptable), np.fmin(acc_high, np.fmax(wri_poor, time_late)))
        
        habits_good = np.fmax(wri_good, time_early)
        high_avg_cond = np.fmax(np.fmin(acc_high, np.fmax(wri_avg, time_ok)), np.fmin(acc_med, habits_good))

        superior_cond = np.fmin(acc_high, habits_good)
        excellent_cond = np.fmin(acc_high, np.fmin(wri_good, time_early))

        # Aggregation
        out_fail = np.fmin(fail_cond, trapezoidal_left(self.x_out, 60, 75))
        out_pass = np.fmin(pass_cond, triangular_membership(self.x_out, 65, 80, 95))
        out_high_avg = np.fmin(high_avg_cond, triangular_membership(self.x_out, 75, 88, 100))
        out_superior = np.fmin(superior_cond, triangular_membership(self.x_out, 88, 95, 105))
        out_excellent = np.fmin(excellent_cond, trapezoidal_right(self.x_out, 96, 115))

        aggregated = np.fmax(out_fail, np.fmax(out_pass, np.fmax(out_high_avg, np.fmax(out_superior, out_excellent))))
        return aggregated

    def defuzzify(self, aggregated):
        numerator = np.sum(self.x_out * aggregated)
        denominator = np.sum(aggregated)
        if denominator == 0: return 0
        return min(100.0, numerator / denominator)

# --- Main App ---

def main():
    sys = GradingSystem()

    with st.sidebar:
        logo_path = "static/profile_v1.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.markdown("### AI Club")

        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h3>Artificial Intelligence Club</h3>
            <p style='font-size: 0.9em; color: #666;'>
            <b>Project:</b> Fuzzy Grader<br>
            <b>Dev:</b> Gem Christian O. Lazo<br>
            <b>Prof:</b> Jan Eilbert Lee
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        
        with st.expander("⚙️ Grading Calibration", expanded=False):
            st.markdown("Adjust the strictness of the system.")
            calib_acc_high = st.slider("Accuracy Strictness", 50, 85, 65)
            calib_wri_good = st.slider("Writing Strictness", 3.0, 8.0, 5.0, 0.5)
            calib_time_good = st.slider("Deadline Strictness", 3.0, 8.0, 5.0, 0.5)
            
            calibration_params = {
                'acc_high_start': calib_acc_high,
                'wri_good_start': calib_wri_good,
                'time_good_start': calib_time_good
            }

    st.title("Fuzzy Logic Paper Grader")
    st.markdown("### Grading System (5-Point Scale)")

    tab1, tab2 = st.tabs(["Single Student Grading", "Batch Class Grading (CSV)"])

    # --- TAB 1: SINGLE STUDENT ---
    with tab1:
        st.subheader("Individual Evaluation")
        col_in1, col_in2, col_in3 = st.columns(3)
        options = ["Very Bad", "Poor", "Average", "Good", "Excellent"]
        
        with col_in1:
            txt_acc = st.select_slider("Content Accuracy", options=options, value="Good")
            val_acc = map_input_to_value(txt_acc, 'acc')

        with col_in2:
            txt_wri = st.select_slider("Writing Style", options=options, value="Average")
            val_wri = map_input_to_value(txt_wri, 'wri')

        with col_in3:
            txt_time = st.select_slider("Submission Time", options=options, value="Good")
            val_time = map_input_to_value(txt_time, 'time')

        fuzzed = sys.fuzzify(val_acc, val_wri, val_time, calibration_params)
        aggregated = sys.evaluate_rules(fuzzed)
        final_score = sys.defuzzify(aggregated)
        uni_grade, description, status_color = get_university_grade(final_score)

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.container(border=True)
            st.markdown(f"<h1 style='text-align: center; color: #333;'>{uni_grade}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>{description}</h3>", unsafe_allow_html=True)
            st.metric("Fuzzy Score", f"{final_score:.2f}")
            if status_color == "success": st.success("PASSED")
            elif status_color == "warning": st.warning("PASSED")
            else: st.error("FAILED")

        with res_col2:
            st.caption("Logic Visualization (Defuzzification)")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(sys.x_out, trapezoidal_left(sys.x_out, 60, 75), 'r--', alpha=0.1, label='Fail')
            ax.plot(sys.x_out, triangular_membership(sys.x_out, 65, 80, 95), 'y--', alpha=0.1, label='Pass')
            ax.plot(sys.x_out, triangular_membership(sys.x_out, 88, 95, 105), 'b--', alpha=0.1, label='Superior')
            ax.plot(sys.x_out, trapezoidal_right(sys.x_out, 96, 115), 'g-', alpha=0.1, label='Excellent')
            ax.fill_between(sys.x_out, aggregated, color='#636EFA', alpha=0.4)
            ax.plot(sys.x_out, aggregated, 'k', linewidth=1)
            ax.vlines(final_score, 0, 1, colors='red', linestyles='dashed')
            ax.set_yticks([])
            ax.set_xlim(0, 100)
            st.pyplot(fig)

        st.divider()
        st.subheader("Logic Landscape Analysis")
        st.markdown("This 3D Surface map shows how the AI makes decisions across the entire range.")
        
        # 3D PLOT RESTORED
        plot_col1, plot_col2 = st.columns([1, 3])
        with plot_col1:
             plot_mode = st.radio("Select Axis View", ["Accuracy vs Writing", "Accuracy vs Timeliness", "Writing vs Timeliness"])
             st.info("Rotate the graph to see the score plateaus.")
        
        with plot_col2:
            res = 20
            if plot_mode == "Accuracy vs Writing":
                x_range, y_range = np.linspace(0, 100, res), np.linspace(0, 10, res)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)
                x_lbl, y_lbl = "Accuracy", "Writing"
                for i in range(res):
                    for j in range(res):
                        agg = sys.evaluate_rules(sys.fuzzify(X[i,j], Y[i,j], val_time, calibration_params))
                        Z[i,j] = sys.defuzzify(agg)

            elif plot_mode == "Accuracy vs Timeliness":
                x_range, y_range = np.linspace(0, 100, res), np.linspace(0, 10, res)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)
                x_lbl, y_lbl = "Accuracy", "Timeliness"
                for i in range(res):
                    for j in range(res):
                        agg = sys.evaluate_rules(sys.fuzzify(X[i,j], val_wri, Y[i,j], calibration_params))
                        Z[i,j] = sys.defuzzify(agg)
            else:
                x_range, y_range = np.linspace(0, 10, res), np.linspace(0, 10, res)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)
                x_lbl, y_lbl = "Writing", "Timeliness"
                for i in range(res):
                    for j in range(res):
                        agg = sys.evaluate_rules(sys.fuzzify(val_acc, X[i,j], Y[i,j], calibration_params))
                        Z[i,j] = sys.defuzzify(agg)

            fig_plot = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn', opacity=0.9)])
            fig_plot.update_layout(
                scene = dict(xaxis_title=x_lbl, yaxis_title=y_lbl, zaxis_title='Final Score'),
                margin=dict(l=0, r=0, b=0, t=0),
                height=500
            )
            st.plotly_chart(fig_plot, use_container_width=True)

    # --- TAB 2: BATCH CLASS GRADING ---
    with tab2:
        st.subheader("Batch Processing (Class Mode)")
        st.markdown("Upload a CSV file. Columns should be: `Name`, `Accuracy`, `Writing`, `Timeliness`.")
        
        uploaded_file = st.file_uploader("Upload Class CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['Name', 'Accuracy', 'Writing', 'Timeliness']
                
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {required_cols}")
                else:
                    st.success(f"Loaded {len(df)} students.")
                    st.dataframe(df.head())
                    
                    # --- BUTTON TO RUN GRADING ---
                    if st.button("Run Auto-Grader for All"):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for index, row in df.iterrows():
                            # 1. Convert Text to Numbers
                            n_acc = map_input_to_value(row['Accuracy'], 'acc')
                            n_wri = map_input_to_value(row['Writing'], 'wri')
                            n_time = map_input_to_value(row['Timeliness'], 'time')

                            # 2. Calculate Logic
                            f = sys.fuzzify(n_acc, n_wri, n_time, calibration_params)
                            agg = sys.evaluate_rules(f)
                            score = sys.defuzzify(agg)
                            grade, desc, status = get_university_grade(score)
                            
                            # 3. Pack Data for Export
                            results.append({
                                "Name": row['Name'],
                                "Input Acc": row['Accuracy'],
                                "Input Wri": row['Writing'],
                                "Input Time": row['Timeliness'],
                                "Calculated Score": round(score, 2), 
                                "Final Grade": grade,                
                                "Rating": desc,                      
                                "Status": status.upper()             
                            })
                            progress_bar.progress((index + 1) / len(df))
                        
                        # 4. Create Results DataFrame
                        result_df = pd.DataFrame(results)
                        
                        st.success("Grading Complete!")
                        
                        # Show the table with colors
                        st.dataframe(result_df.style.applymap(
                            lambda v: 'color: red;' if v == 'FAILED' else 'color: green;', subset=['Status']
                        ))
                        
                        # --- EXPORT SECTION ---
                        st.divider()
                        st.subheader("📥 Export Results")
                        st.markdown("Download the fully graded class record below:")
                        
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="Download Graded_Class_Results.csv",
                            data=csv_data,
                            file_name="Graded_Class_Results.csv",
                            mime="text/csv",
                            key='download-csv'
                        )
                        
                        # Show Statistics
                        st.divider()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Class Average", f"{result_df['Calculated Score'].mean():.2f}")
                        pass_count = len(result_df[result_df['Status']!='FAILED'])
                        c2.metric("Pass Rate", f"{(pass_count / len(df) * 100):.1f}%")
                        c3.metric("Highest Score", f"{result_df['Calculated Score'].max()}")

            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        else:
            st.info("Don't have a file? Download this template.")
            example_data = pd.DataFrame([
                {"Name": "Student A", "Accuracy": "Excellent", "Writing": "Good", "Timeliness": "Average"},
                {"Name": "Student B", "Accuracy": "Poor", "Writing": "Very Bad", "Timeliness": "Very Bad"},
                {"Name": "Student C", "Accuracy": "Good", "Writing": "Average", "Timeliness": "Good"}
            ])
            st.dataframe(example_data)
            st.download_button("Download Template CSV", example_data.to_csv(index=False).encode('utf-8'), "template.csv")

if __name__ == "__main__":
    main()