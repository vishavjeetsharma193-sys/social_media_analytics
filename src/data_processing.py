import pandas as pd
import numpy as np
import os

def generate_temporal_dataset(
    raw_filepath="data/students_temporal_usage.csv", 
    processed_filepath="data/students_behavioral_change_features.csv",
    num_students=800, 
    seed=42
):
    """
    Generates longitudinal temporal social media usage data across 4 weeks for each student.
    Extracts personal baseline modeling and behavioral change (Delta) metrics.
    """
    np.random.seed(seed)
    
    genders = np.random.choice(['Male', 'Female'], size=num_students, p=[0.51, 0.49])
    ages = np.random.choice(range(16, 25), size=num_students)
    
    def get_academic_level(age):
        if age <= 18:
            return 'High School'
        elif age <= 21:
            return 'Undergraduate'
        else:
            return 'Graduate'
            
    academic_levels = [get_academic_level(a) for a in ages]
    platforms = np.random.choice(
        ['Instagram', 'TikTok', 'Snapchat', 'YouTube', 'Facebook', 'WhatsApp', 'Twitter', 'LinkedIn'],
        size=num_students,
        p=[0.34, 0.28, 0.12, 0.10, 0.07, 0.04, 0.03, 0.02]
    )
    
    # Archetype trajectories:
    # 0: Stable / low risk (40%)
    # 1: Increasing risk (30%)
    # 2: Persistent high risk (15%)
    # 3: Rapidly increasing risk (15%)
    archetypes = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.40, 0.30, 0.15, 0.15])
    
    temporal_records = []
    summary_features = []
    
    for i in range(num_students):
        student_id = f"STU_{1000+i}"
        gender = genders[i]
        age = ages[i]
        academic = academic_levels[i]
        platform = platforms[i]
        arch = archetypes[i]
        
        # Base initial usage (Week 1)
        if arch == 0: # Stable low
            base_screen_time = np.random.uniform(1.5, 3.5)
            base_sessions = np.random.randint(10, 20)
            base_avg_dur = np.random.uniform(8.0, 15.0)
            base_night = np.random.uniform(0.1, 0.8)
            base_check_freq = np.random.uniform(2.0, 5.0)
            base_sleep_use = np.random.uniform(0.1, 0.6)
            growth_factor = np.random.uniform(-0.05, 0.05)
            noise_scale = 0.3
        elif arch == 1: # Increasing risk
            base_screen_time = np.random.uniform(2.5, 4.0)
            base_sessions = np.random.randint(15, 25)
            base_avg_dur = np.random.uniform(10.0, 18.0)
            base_night = np.random.uniform(0.5, 1.5)
            base_check_freq = np.random.uniform(4.0, 8.0)
            base_sleep_use = np.random.uniform(0.4, 1.0)
            growth_factor = np.random.uniform(0.3, 0.7)
            noise_scale = 0.4
        elif arch == 2: # Persistent high
            base_screen_time = np.random.uniform(5.5, 7.5)
            base_sessions = np.random.randint(28, 45)
            base_avg_dur = np.random.uniform(18.0, 30.0)
            base_night = np.random.uniform(1.8, 3.2)
            base_check_freq = np.random.uniform(9.0, 16.0)
            base_sleep_use = np.random.uniform(1.2, 2.2)
            growth_factor = np.random.uniform(0.02, 0.2)
            noise_scale = 0.5
        else: # Rapidly increasing risk (Steep escalation!)
            base_screen_time = np.random.uniform(2.0, 3.5)
            base_sessions = np.random.randint(12, 22)
            base_avg_dur = np.random.uniform(9.0, 16.0)
            base_night = np.random.uniform(0.4, 1.2)
            base_check_freq = np.random.uniform(3.5, 7.0)
            base_sleep_use = np.random.uniform(0.3, 0.9)
            growth_factor = np.random.uniform(0.9, 1.5) # Fast surge
            noise_scale = 0.6
            
        # Simulate 4 weeks of metrics
        weeks_screen_time = []
        weeks_sessions = []
        weeks_avg_dur = []
        weeks_night = []
        weeks_check = []
        weeks_sleep = []
        
        for w in range(4):
            # w = 0, 1, 2, 3 (Week 1, Week 2, Week 3, Week 4)
            st = base_screen_time + (w * growth_factor) + np.random.normal(0, noise_scale)
            st = np.clip(st, 0.5, 12.0)
            
            sess = base_sessions + int(w * growth_factor * 4) + np.random.randint(-2, 3)
            sess = max(5, sess)
            
            dur = base_avg_dur + (w * growth_factor * 2.0) + np.random.normal(0, 1.0)
            dur = np.clip(dur, 4.0, 45.0)
            
            night = base_night + (w * growth_factor * 0.45) + np.random.normal(0, 0.15)
            night = np.clip(night, 0.0, 5.0)
            
            check = base_check_freq + (w * growth_factor * 2.2) + np.random.normal(0, 0.5)
            check = np.clip(check, 1.0, 25.0)
            
            slp = base_sleep_use + (w * growth_factor * 0.3) + np.random.normal(0, 0.1)
            slp = np.clip(slp, 0.0, 3.0)
            
            weeks_screen_time.append(st)
            weeks_sessions.append(sess)
            weeks_avg_dur.append(dur)
            weeks_night.append(night)
            weeks_check.append(check)
            weeks_sleep.append(slp)
            
            temporal_records.append({
                'Student_ID': student_id,
                'Week': w + 1,
                'Screen_Time_Hours': np.round(st, 2),
                'Session_Frequency': int(sess),
                'Avg_Session_Duration_Mins': np.round(dur, 1),
                'Night_Time_Usage_Hours': np.round(night, 2),
                'Checking_Frequency_PerHour': np.round(check, 1),
                'Sleep_Related_Usage_Hours': np.round(slp, 2)
            })
            
        # Personal Baseline (Week 1 or mean of initial period)
        base_st = weeks_screen_time[0]
        base_sess = weeks_sessions[0]
        base_dur = weeks_avg_dur[0]
        base_ngt = weeks_night[0]
        base_chk = weeks_check[0]
        base_slp = weeks_sleep[0]
        
        # Current Week (Week 4)
        curr_st = weeks_screen_time[3]
        curr_sess = weeks_sessions[3]
        curr_dur = weeks_avg_dur[3]
        curr_ngt = weeks_night[3]
        curr_chk = weeks_check[3]
        curr_slp = weeks_sleep[3]
        
        # Behavioral Change Detection (Deltas relative to personal baseline)
        delta_st = curr_st - base_st
        delta_sess = curr_sess - base_sess
        delta_dur = curr_dur - base_dur
        delta_ngt = curr_ngt - base_ngt
        delta_chk = curr_chk - base_chk
        delta_slp = curr_slp - base_slp
        usage_variability = np.std(weeks_screen_time)
        
        # Simulated GPA impact (correlated with final screen time & night usage)
        gpa = 3.9 - (curr_st * 0.15) - (curr_ngt * 0.25) + np.random.normal(0, 0.25)
        gpa = np.clip(gpa, 1.8, 4.0)
        
        # True Addiction Risk Ground Truth (derived from trajectory escalation + current intensity)
        risk_score_raw = (
            (curr_st * 0.25) + 
            (curr_ngt * 0.40) + 
            (delta_st * 0.60) + 
            (delta_ngt * 0.75) + 
            (delta_chk * 0.15) + 
            (usage_variability * 0.5)
        )
        # Normalize target probability between 0 and 1
        risk_prob_true = 1 / (1 + np.exp(-(risk_score_raw - 4.5)))
        high_risk_flag = 1 if risk_prob_true > 0.55 else 0
        
        summary_features.append({
            'Student_ID': student_id,
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic,
            'Most_Used_Platform': platform,
            'GPA': np.round(gpa, 2),
            'Archetype_True': arch,
            # Current absolute usage metrics
            'Curr_Screen_Time': np.round(curr_st, 2),
            'Curr_Session_Freq': int(curr_sess),
            'Curr_Avg_Session_Dur': np.round(curr_dur, 1),
            'Curr_Night_Usage': np.round(curr_ngt, 2),
            'Curr_Checking_Freq': np.round(curr_chk, 1),
            'Curr_Sleep_Usage': np.round(curr_slp, 2),
            # Personal Baseline metrics (Week 1)
            'Base_Screen_Time': np.round(base_st, 2),
            'Base_Session_Freq': int(base_sess),
            'Base_Avg_Session_Dur': np.round(base_dur, 1),
            'Base_Night_Usage': np.round(base_ngt, 2),
            'Base_Checking_Freq': np.round(base_chk, 1),
            'Base_Sleep_Usage': np.round(base_slp, 2),
            # Behavioral Change Deltas (Key Novelty Features)
            'Delta_Screen_Time': np.round(delta_st, 2),
            'Delta_Session_Freq': np.round(delta_sess, 2),
            'Delta_Avg_Session_Dur': np.round(delta_dur, 1),
            'Delta_Night_Usage': np.round(delta_ngt, 2),
            'Delta_Checking_Freq': np.round(delta_chk, 1),
            'Delta_Sleep_Usage': np.round(delta_slp, 2),
            'Usage_Variability': np.round(usage_variability, 2),
            # Trajectory values for visual plotting (W1..W4)
            'W1_Screen': np.round(weeks_screen_time[0], 2),
            'W2_Screen': np.round(weeks_screen_time[1], 2),
            'W3_Screen': np.round(weeks_screen_time[2], 2),
            'W4_Screen': np.round(weeks_screen_time[3], 2),
            'High_Risk_Target': high_risk_flag,
            'True_Risk_Prob': np.round(risk_prob_true, 3)
        })
        
    df_temporal = pd.DataFrame(temporal_records)
    df_features = pd.DataFrame(summary_features)
    
    os.makedirs(os.path.dirname(raw_filepath), exist_ok=True)
    df_temporal.to_csv(raw_filepath, index=False)
    df_features.to_csv(processed_filepath, index=False)
    
    print(f"Generated temporal trace dataset: {raw_filepath} ({len(df_temporal)} rows)")
    print(f"Generated behavioral change features dataset: {processed_filepath} ({len(df_features)} rows)")
    
    return df_temporal, df_features

def preprocess_data(df_features=None):
    """
    Ensures dataset presence and returns behavioral feature dataframe.
    """
    processed_filepath = "data/students_behavioral_change_features.csv"
    raw_filepath = "data/students_temporal_usage.csv"
    
    if df_features is None:
        if not os.path.exists(processed_filepath):
            _, df_features = generate_temporal_dataset(raw_filepath, processed_filepath)
        else:
            df_features = pd.read_csv(processed_filepath)
            
    return df_features

if __name__ == "__main__":
    generate_temporal_dataset()
