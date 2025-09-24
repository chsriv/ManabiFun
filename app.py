import streamlit as st
import json
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time

# Page config
st.set_page_config(
    page_title="ManabiFun - AI English Learning",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for island theme
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .island-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .island-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .stats-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .quiz-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    .question-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #007bff;
        margin: 20px 0;
    }
    .result-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(40, 167, 69, 0.3);
        margin: 20px 0;
    }
    .weak-topic-card {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(255, 193, 7, 0.3);
    }
    .analytics-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 123, 255, 0.4);
    }
    .progress-text {
        font-size: 18px;
        font-weight: bold;
        color: #495057;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'total_xp': 0,
            'streak': 1,
            'completed_islands': [],
            'current_scores': {
                'grammar': [],
                'articles': [],
                'synonyms': [],
                'antonyms': [],
                'sentences': []
            },
            'quiz_start_time': None
        }
    
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None
        
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
        
    if 'current_question_idx' not in st.session_state:
        st.session_state.current_question_idx = 0
        
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []

# Load data functions
@st.cache_data
def load_questions():
    """Load questions from CSV file"""
    try:
        df = pd.read_csv('data/manabifun_questions.csv')
        
        # Convert CSV to dictionary format for easy access
        questions_db = {}
        
        for topic in df['topic'].unique():
            topic_questions = df[df['topic'] == topic]
            questions_db[topic] = []
            
            for _, row in topic_questions.iterrows():
                question_dict = {
                    'question': row['question'],
                    'options': [row['option_a'], row['option_b'], row['option_c'], row['option_d']],
                    'answer': ['A', 'B', 'C', 'D'].index(row['correct_answer']),  # Convert A,B,C,D to 0,1,2,3
                    'difficulty': row['difficulty']
                }
                questions_db[topic].append(question_dict)
        
        return questions_db
        
    except FileNotFoundError:
        st.error("❌ Questions database not found! Please ensure 'data/manabifun_questions.csv' exists.")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading questions: {e}")
        return {}

@st.cache_data
def load_student_scores():
    """Load student scores from CSV"""
    try:
        return pd.read_csv('data/student_scores.csv')
    except FileNotFoundError:
        # Create empty DataFrame if file doesn't exist
        columns = [
            'student_id', 'student_name', 'timestamp', 'quiz_type', 'topic', 
            'score', 'total_questions', 'correct_answers', 'time_spent_minutes',
            'difficulty_level', 'xp_earned', 'streak_day'
        ]
        return pd.DataFrame(columns=columns)
    except Exception as e:
        st.error(f"❌ Error loading student scores: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_ml_model():
    """Load the trained ML model"""
    try:
        with open('models/weakness_detector.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ ML model not found! Please run train_model.py first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading ML model: {e}")
        return None

def save_quiz_result(student_name, quiz_info, score, correct_answers, time_spent, xp_earned):
    """Save quiz result to CSV database"""
    try:
        # Load existing scores
        try:
            scores_df = pd.read_csv('data/student_scores.csv')
        except FileNotFoundError:
            # Create new DataFrame if file doesn't exist
            columns = [
                'student_id', 'student_name', 'timestamp', 'quiz_type', 'topic', 
                'score', 'total_questions', 'correct_answers', 'time_spent_minutes',
                'difficulty_level', 'xp_earned', 'streak_day'
            ]
            scores_df = pd.DataFrame(columns=columns)
        
        # Create student ID if new student
        student_id = f"student_{student_name.lower().replace(' ', '_').replace('-', '_')}"
        
        # Create new record
        new_record = {
            'student_id': student_id,
            'student_name': student_name,
            'timestamp': datetime.now(),
            'quiz_type': quiz_info['type'],
            'topic': quiz_info['topic'],
            'score': score,
            'total_questions': 10,
            'correct_answers': correct_answers,
            'time_spent_minutes': time_spent,
            'difficulty_level': 'mixed',
            'xp_earned': xp_earned,
            'streak_day': st.session_state.user_profile['streak']
        }
        
        # Add new record to DataFrame
        new_record_df = pd.DataFrame([new_record])
        scores_df = pd.concat([scores_df, new_record_df], ignore_index=True)
        
        # Save back to CSV
        scores_df.to_csv('data/student_scores.csv', index=False)
        
        # Clear cache so new data is loaded
        st.cache_data.clear()
        
    except Exception as e:
        st.error(f"❌ Error saving quiz result: {e}")

def predict_weakness(scores_dict):
    """Predict student's weakest area using ML model"""
    model = load_ml_model()
    if model is None:
        return "Grammar Basics", 0.5
    
    try:
        # Calculate average scores for each topic
        avg_scores = []
        topics = ['grammar', 'articles', 'synonyms', 'antonyms', 'sentences']
        
        for topic in topics:
            if scores_dict[topic]:
                avg_scores.append(np.mean(scores_dict[topic]))
            else:
                avg_scores.append(50)  # Default score
        
        # Add time_spent feature (simulated based on performance)
        time_spent = 20 + (100 - np.mean(avg_scores)) * 0.2  # More time if struggling
        features = avg_scores + [time_spent]
        
        # Predict weakness
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0].max()
        
        topic_names = ['Grammar Basics', 'Articles', 'Synonyms', 'Antonyms', 'Sentence Formation']
        return topic_names[prediction], confidence
    
    except Exception as e:
        st.error(f"❌ Error predicting weakness: {e}")
        return "Grammar Basics", 0.5

def create_performance_charts(student_name):
    """Create pie charts and performance analytics"""
    scores_df = load_student_scores()
    
    if scores_df.empty:
        st.info("📊 Complete some quizzes to see your performance analysis!")
        return
    
    # Filter data for current student
    student_data = scores_df[scores_df['student_name'] == student_name]
    
    if student_data.empty:
        st.info("📊 Complete some quizzes to see your performance analysis!")
        return
    
    if len(student_data) < 2:
        st.info("📊 Complete at least 2 quizzes to see detailed analysis!")
        return
    
    st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by topic pie chart
        topic_performance = student_data.groupby('topic')['score'].mean()
        
        fig_pie = px.pie(
            values=topic_performance.values,
            names=[name.replace('_', ' ').title() for name in topic_performance.index],
            title="📊 Performance by Topic (Average Scores)",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        )
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12
        )
        fig_pie.update_layout(
            height=400,
            font=dict(size=14),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        # Weekly progress line chart
        student_data['date'] = pd.to_datetime(student_data['timestamp']).dt.date
        daily_progress = student_data.groupby('date')['score'].mean().reset_index()
        
        fig_line = px.line(
            daily_progress,
            x='date',
            y='score',
            title="📈 Daily Progress Trend",
            labels={'date': 'Date', 'score': 'Average Score (%)'},
            markers=True,
            line_shape='spline'
        )
        fig_line.update_traces(
            line=dict(color='#007bff', width=3),
            marker=dict(size=8, color='#007bff')
        )
        fig_line.update_layout(
            height=400,
            font=dict(size=14),
            xaxis_title="Date",
            yaxis_title="Score (%)",
            hovermode='x'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Strengths and Weaknesses Analysis
    st.markdown("### 🎯 Detailed Performance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Strongest topic
        strongest_topic = topic_performance.idxmax()
        strongest_score = topic_performance.max()
        st.markdown(f"""
        <div class="result-card">
            <h3>💪 Strongest Topic</h3>
            <h2>{strongest_topic.replace('_', ' ').title()}</h2>
            <p style="font-size: 24px; font-weight: bold;">{strongest_score:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Weakest topic  
        weakest_topic = topic_performance.idxmin()
        weakest_score = topic_performance.min()
        st.markdown(f"""
        <div class="weak-topic-card">
            <h3>📈 Needs Improvement</h3>
            <h2>{weakest_topic.replace('_', ' ').title()}</h2>
            <p style="font-size: 24px; font-weight: bold;">{weakest_score:.1f}%</p>
            <p>💡 Practice this topic more!</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Overall average
        overall_avg = student_data['score'].mean()
        if overall_avg >= 80:
            grade_emoji = "🌟"
            grade_color = "#28a745"
        elif overall_avg >= 70:
            grade_emoji = "⭐"
            grade_color = "#17a2b8"
        elif overall_avg >= 60:
            grade_emoji = "💫"
            grade_color = "#ffc107"
        else:
            grade_emoji = "📚"
            grade_color = "#6c757d"
            
        st.markdown(f"""
        <div class="analytics-card" style="background: {grade_color}; color: white; text-align: center;">
            <h3>{grade_emoji} Overall Average</h3>
            <h1 style="margin: 0;">{overall_avg:.1f}%</h1>
            <p>Keep up the great work!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown bar chart
    fig_bar = px.bar(
        x=[name.replace('_', ' ').title() for name in topic_performance.index],
        y=topic_performance.values,
        title="📊 Detailed Topic Performance Breakdown",
        labels={'x': 'Topics', 'y': 'Average Score (%)'},
        color=topic_performance.values,
        color_continuous_scale='RdYlGn',
        text=topic_performance.values
    )
    fig_bar.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        textfont_size=14
    )
    fig_bar.update_layout(
        coloraxis_showscale=False,
        height=400,
        font=dict(size=14),
        xaxis_title="Learning Topics",
        yaxis_title="Average Score (%)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recent quiz history
    st.markdown("### 📋 Recent Quiz History")
    recent_quizzes = student_data.tail(10)[['timestamp', 'topic', 'score', 'xp_earned', 'time_spent_minutes']].copy()
    recent_quizzes['timestamp'] = pd.to_datetime(recent_quizzes['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    recent_quizzes['topic'] = recent_quizzes['topic'].str.replace('_', ' ').str.title()
    recent_quizzes.columns = ['Date & Time', 'Topic', 'Score (%)', 'XP Earned', 'Time (min)']
    
    # Style the dataframe
    def color_score(val):
        if val >= 80:
            return 'background-color: #d4edda; color: #155724'
        elif val >= 70:
            return 'background-color: #d1ecf1; color: #0c5460'
        elif val >= 60:
            return 'background-color: #fff3cd; color: #856404'
        else:
            return 'background-color: #f8d7da; color: #721c24'
    
    styled_df = recent_quizzes.style.applymap(color_score, subset=['Score (%)'])
    st.dataframe(styled_df, use_container_width=True)

def main_dashboard():
    """Main dashboard with islands"""
    st.markdown(f"""
    <div class="main-header">
        <h1>🏝️ Welcome to ManabiFun, {st.session_state.user_profile['name']}!</h1>
        <p>Your AI-Powered English Learning Adventure Awaits</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h2>🌟</h2>
            <h3>Total XP</h3>
            <h1>{st.session_state.user_profile['total_xp']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h2>🔥</h2>
            <h3>Learning Streak</h3>
            <h1>{st.session_state.user_profile['streak']} day{'s' if st.session_state.user_profile['streak'] != 1 else ''}</h1>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        completed = len(st.session_state.user_profile['completed_islands'])
        st.markdown(f"""
        <div class="stats-card">
            <h2>🏆</h2>
            <h3>Islands Completed</h3>
            <h1>{completed}/5</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Islands selection
    st.markdown("## 🗺️ Choose Your Learning Island")
    
    islands = [
        {"name": "Grammar Basics Island", "emoji": "🏝️", "topic": "grammar", "description": "Master basic grammar rules"},
        {"name": "Articles Island", "emoji": "🏖️", "topic": "articles", "description": "Learn a, an, the"},
        {"name": "Synonyms Island", "emoji": "🌴", "topic": "synonyms", "description": "Discover similar words"},
        {"name": "Antonyms Island", "emoji": "🏛️", "topic": "antonyms", "description": "Explore opposite words"},
        {"name": "Sentence Formation Island", "emoji": "🏰", "topic": "sentences", "description": "Build perfect sentences"}
    ]
    
    # Display islands in a grid
    cols = st.columns(3)
    for i, island in enumerate(islands):
        with cols[i % 3]:
            is_completed = island['topic'] in st.session_state.user_profile['completed_islands']
            status_emoji = "✅" if is_completed else "🎯"
            
            button_text = f"{island['emoji']} {island['name']} {status_emoji}"
            if st.button(button_text, key=f"island_{i}", use_container_width=True, help=island['description']):
                start_quiz(island['topic'], island['name'])
    
    # Weekly Challenge Ship (unlocked after completing 2+ islands)
    if len(st.session_state.user_profile['completed_islands']) >= 2:
        st.markdown("---")
        st.markdown("## 🚢 Weekly Challenge Ship Unlocked!")
        st.markdown("*Test your knowledge with mixed questions from completed islands*")
        if st.button("🚢 Take Weekly Mixed Quiz Challenge", use_container_width=True):
            start_mixed_quiz()
    
    # Performance Analytics Dashboard
    if st.session_state.user_profile['name']:
        st.markdown("---")
        st.markdown("## 📊 Your Performance Analytics")
        create_performance_charts(st.session_state.user_profile['name'])

def start_quiz(topic, island_name):
    """Start a quiz for specific topic"""
    questions_db = load_questions()
    if not questions_db or topic not in questions_db:
        st.error(f"❌ No questions found for {topic}")
        return
    
    # Select 10 random questions
    all_questions = questions_db[topic]
    if len(all_questions) < 10:
        st.warning(f"⚠️ Only {len(all_questions)} questions available for {topic}")
        selected_questions = all_questions
    else:
        selected_questions = random.sample(all_questions, 10)
    
    st.session_state.current_quiz = {
        'topic': topic,
        'island_name': island_name,
        'type': 'single_topic'
    }
    st.session_state.quiz_questions = selected_questions
    st.session_state.current_question_idx = 0
    st.session_state.user_answers = []
    st.session_state.quiz_start_time = time.time()
    
    st.rerun()

def start_mixed_quiz():
    """Start mixed quiz from completed islands"""
    questions_db = load_questions()
    completed_topics = st.session_state.user_profile['completed_islands']
    
    if len(completed_topics) < 2:
        st.error("❌ Complete at least 2 islands to unlock mixed quiz!")
        return
    
    if not questions_db:
        st.error("❌ Questions database not available!")
        return
    
    # Mix questions from completed topics
    mixed_questions = []
    questions_per_topic = max(1, 10 // len(completed_topics))  # Distribute 10 questions across topics
    
    for topic in completed_topics:
        if topic in questions_db and questions_db[topic]:
            topic_questions = random.sample(
                questions_db[topic], 
                min(questions_per_topic, len(questions_db[topic]))
            )
            mixed_questions.extend(topic_questions)
    
    # Ensure we have exactly 10 questions
    if len(mixed_questions) > 10:
        selected_questions = random.sample(mixed_questions, 10)
    elif len(mixed_questions) < 10:
        # Add more questions if needed
        remaining_needed = 10 - len(mixed_questions)
        for topic in completed_topics:
            if remaining_needed <= 0:
                break
            if topic in questions_db:
                available_questions = [q for q in questions_db[topic] if q not in mixed_questions]
                additional_questions = random.sample(
                    available_questions, 
                    min(remaining_needed, len(available_questions))
                )
                mixed_questions.extend(additional_questions)
                remaining_needed -= len(additional_questions)
        selected_questions = mixed_questions
    else:
        selected_questions = mixed_questions
    
    st.session_state.current_quiz = {
        'topic': 'mixed',
        'island_name': 'Weekly Challenge Ship',
        'type': 'mixed'
    }
    st.session_state.quiz_questions = selected_questions
    st.session_state.current_question_idx = 0
    st.session_state.user_answers = []
    st.session_state.quiz_start_time = time.time()
    
    st.rerun()

def quiz_interface():
    """Display quiz interface"""
    quiz_info = st.session_state.current_quiz
    questions = st.session_state.quiz_questions
    current_idx = st.session_state.current_question_idx
    
    if current_idx >= len(questions):
        show_quiz_results()
        return
    
    current_q = questions[current_idx]
    
    st.markdown(f"""
    <div class="quiz-container">
        <h1>{quiz_info['island_name']} 🎯</h1>
        <div class="progress-text">Question {current_idx + 1} of {len(questions)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    progress = (current_idx) / len(questions)
    st.progress(progress)
    
    # Question card
    st.markdown(f"""
    <div class="question-card">
        <h2>❓ {current_q['question']}</h2>
        <p><strong>Difficulty:</strong> <span style="color: {'#28a745' if current_q['difficulty'] == 'easy' else '#ffc107' if current_q['difficulty'] == 'medium' else '#dc3545'};">{current_q['difficulty'].title()}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    st.markdown("### Choose your answer:")
    selected_option = st.radio(
        "Options:",
        range(len(current_q['options'])),
        format_func=lambda x: f"{chr(65+x)}. {current_q['options'][x]}",
        key=f"q_{current_idx}",
        label_visibility="collapsed"
    )
    
    # Navigation
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if current_idx > 0:
            if st.button("⬅️ Previous Question", use_container_width=True):
                st.session_state.current_question_idx -= 1
                st.rerun()
    
    with col2:
        if st.button("✅ Submit Answer", use_container_width=True, type="primary"):
            # Store answer
            if len(st.session_state.user_answers) <= current_idx:
                st.session_state.user_answers.extend([None] * (current_idx + 1 - len(st.session_state.user_answers)))
            st.session_state.user_answers[current_idx] = selected_option
            
            st.session_state.current_question_idx += 1
            st.rerun()
    
    with col3:
        if current_idx < len(questions) - 1:
            if st.button("➡️ Next Question", use_container_width=True):
                # Store current answer before moving
                if len(st.session_state.user_answers) <= current_idx:
                    st.session_state.user_answers.extend([None] * (current_idx + 1 - len(st.session_state.user_answers)))
                st.session_state.user_answers[current_idx] = selected_option
                
                st.session_state.current_question_idx += 1
                st.rerun()
    
    with col4:
        if st.button("🏠 Back to Islands", use_container_width=True):
            st.session_state.current_quiz = None
            st.rerun()
    
    # Show current answers summary
    if st.session_state.user_answers:
        with st.expander(f"📝 Your Answers So Far ({len([a for a in st.session_state.user_answers if a is not None])}/{len(questions)})", expanded=False):
            for i, answer in enumerate(st.session_state.user_answers):
                if answer is not None and i < len(questions):
                    st.write(f"Q{i+1}: {chr(65+answer)}. {questions[i]['options'][answer]}")

def show_quiz_results():
    """Show quiz results and update user profile"""
    quiz_info = st.session_state.current_quiz
    questions = st.session_state.quiz_questions
    user_answers = st.session_state.user_answers
    
    # Calculate score
    correct_answers = 0
    detailed_results = []
    
    for i, question in enumerate(questions):
        user_answer = user_answers[i] if i < len(user_answers) else None
        is_correct = user_answer == question['answer']
        if is_correct:
            correct_answers += 1
        
        detailed_results.append({
            'question': question['question'],
            'user_answer': question['options'][user_answer] if user_answer is not None else "No answer",
            'correct_answer': question['options'][question['answer']],
            'is_correct': is_correct,
            'difficulty': question['difficulty']
        })
    
    score_percentage = (correct_answers / len(questions)) * 100
    xp_earned = correct_answers * 10  # 10 XP per correct answer
    
    # Calculate time spent
    if st.session_state.quiz_start_time:
        time_spent = (time.time() - st.session_state.quiz_start_time) / 60  # Convert to minutes
        time_spent = max(1, round(time_spent))  # At least 1 minute
    else:
        time_spent = np.random.randint(5, 15)  # Fallback random time
    
    # Update user profile
    st.session_state.user_profile['total_xp'] += xp_earned
    
    # Save to CSV database
    save_quiz_result(
        st.session_state.user_profile['name'],
        quiz_info,
        score_percentage,
        correct_answers,
        time_spent,
        xp_earned
    )
    
    if quiz_info['type'] == 'single_topic':
        topic = quiz_info['topic']
        st.session_state.user_profile['current_scores'][topic].append(score_percentage)
        
        # Mark island as completed if score >= 70%
        if score_percentage >= 70 and topic not in st.session_state.user_profile['completed_islands']:
            st.session_state.user_profile['completed_islands'].append(topic)
            # Bonus XP for completing island
            st.session_state.user_profile['total_xp'] += 50
            st.balloons()  # Celebration!
    
    # Display results
    st.markdown("## 🎉 Quiz Completed!")
    
    # Results summary card
    if score_percentage >= 90:
        result_color = "#28a745"
        result_emoji = "🌟"
        result_message = "Outstanding! You're a superstar!"
    elif score_percentage >= 80:
        result_color = "#17a2b8"
        result_emoji = "⭐"
        result_message = "Excellent work! Keep it up!"
    elif score_percentage >= 70:
        result_color = "#ffc107"
        result_emoji = "💫"
        result_message = "Great job! You're improving!"
    elif score_percentage >= 60:
        result_color = "#fd7e14"
        result_emoji = "📚"
        result_message = "Good effort! Practice more!"
    else:
        result_color = "#dc3545"
        result_emoji = "💪"
        result_message = "Keep practicing! You'll get better!"
    
    st.markdown(f"""
    <div class="result-card" style="background: linear-gradient(135deg, {result_color} 0%, {result_color}dd 100%);">
        <h1>{result_emoji} Your Results</h1>
        <h2>{correct_answers}/{len(questions)} Correct ({score_percentage:.1f}%)</h2>
        <h3>🌟 {xp_earned} XP Earned</h3>
        <h3>⏱️ Time: {time_spent} minutes</h3>
        <p style="font-size: 18px; margin-top: 20px;">{result_message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Progress gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Quiz Score", 'font': {'size': 24}},
            delta = {'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': result_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#f8d7da'},
                    {'range': [50, 70], 'color': '#fff3cd'},
                    {'range': [70, 90], 'color': '#d1ecf1'},
                    {'range': [90, 100], 'color': '#d4edda'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400, font={'size': 16})
        st.plotly_chart(fig, use_container_width=True)
        
        # Island completion status
        if quiz_info['type'] == 'single_topic':
            topic = quiz_info['topic']
            if score_percentage >= 70:
                if topic in st.session_state.user_profile['completed_islands']:
                    st.success(f"🎉 {quiz_info['island_name']} Completed! (+50 Bonus XP)")
                else:
                    st.success(f"🎯 Island Unlocked: {quiz_info['island_name']}")
            else:
                st.info(f"💪 Score 70%+ to complete {quiz_info['island_name']}")
    
    with col2:
        # Show weakness analysis if enough data
        all_scores = st.session_state.user_profile['current_scores']
        if any(scores for scores in all_scores.values()):
            weak_topic, confidence = predict_weakness(all_scores)
            
            st.markdown(f"""
            <div class="weak-topic-card">
                <h3>🤖 AI Recommendation</h3>
                <p>Based on your performance patterns:</p>
                <h2>Focus on {weak_topic}</h2>
                <p>Confidence: {confidence:.1%}</p>
                <p>💡 <em>Practice this topic to boost your overall score!</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance comparison chart
            topic_avgs = {}
            topic_labels = {
                'grammar': 'Grammar',
                'articles': 'Articles', 
                'synonyms': 'Synonyms',
                'antonyms': 'Antonyms',
                'sentences': 'Sentences'
            }
            
            for topic, scores in all_scores.items():
                if scores:
                    topic_avgs[topic_labels.get(topic, topic.title())] = np.mean(scores)
            
            if len(topic_avgs) >= 2:
                fig = px.radar(
                    r=list(topic_avgs.values()),
                    theta=list(topic_avgs.keys()),
                    title="📊 Your Learning Profile",
                    range_r=[0, 100]
                )
                fig.update_traces(fill='toself', line_color='#007bff')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🎯 Complete more quizzes to get personalized AI recommendations!")
    
    # Detailed question review
    st.markdown("---")
    st.markdown("### 📝 Detailed Question Review")
    
    correct_count = sum(1 for r in detailed_results if r['is_correct'])
    incorrect_count = len(detailed_results) - correct_count
    
    # Summary tabs
    tab1, tab2, tab3 = st.tabs([f"📊 Summary", f"✅ Correct ({correct_count})", f"❌ Review ({incorrect_count})"])
    
    with tab1:
        # Performance by difficulty
        difficulty_performance = {}
        for result in detailed_results:
            diff = result['difficulty']
            if diff not in difficulty_performance:
                difficulty_performance[diff] = {'correct': 0, 'total': 0}
            difficulty_performance[diff]['total'] += 1
            if result['is_correct']:
                difficulty_performance[diff]['correct'] += 1
        
        if difficulty_performance:
            diff_data = []
            for diff, stats in difficulty_performance.items():
                percentage = (stats['correct'] / stats['total']) * 100
                diff_data.append({
                    'Difficulty': diff.title(),
                    'Correct': stats['correct'],
                    'Total': stats['total'],
                    'Percentage': percentage
                })
            
            df_diff = pd.DataFrame(diff_data)
            
            fig_diff = px.bar(
                df_diff,
                x='Difficulty',
                y='Percentage',
                title='Performance by Question Difficulty',
                color='Percentage',
                color_continuous_scale='RdYlGn',
                text='Percentage'
            )
            fig_diff.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_diff.update_layout(coloraxis_showscale=False, height=300)
            st.plotly_chart(fig_diff, use_container_width=True)
            
            st.dataframe(df_diff, use_container_width=True, hide_index=True)
    
    with tab2:
        # Show correct answers
        correct_results = [r for r in detailed_results if r['is_correct']]
        if correct_results:
            for i, result in enumerate(correct_results, 1):
                st.success(f"**Q{i}:** {result['question']}")
                st.write(f"✅ Your answer: {result['user_answer']}")
        else:
            st.info("No correct answers to display.")
    
    with tab3:
        # Show incorrect answers for review
        incorrect_results = [r for r in detailed_results if not r['is_correct']]
        if incorrect_results:
            for i, result in enumerate(incorrect_results, 1):
                st.error(f"**Q{i}:** {result['question']}")
                st.write(f"❌ Your answer: {result['user_answer']}")
                st.write(f"✅ Correct answer: {result['correct_answer']}")
                st.write("---")
        else:
            st.success("🎉 All answers were correct!")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            if quiz_info['type'] == 'single_topic':
                start_quiz(quiz_info['topic'], quiz_info['island_name'])
            else:
                start_mixed_quiz()
    
    with col2:
        if quiz_info['type'] == 'single_topic' and score_percentage < 70:
            if st.button("📚 Practice More", use_container_width=True):
                start_quiz(quiz_info['topic'], quiz_info['island_name'])
        else:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.current_quiz = None
                st.rerun()
    
    with col3:
        if st.button("🏠 Back to Islands", use_container_width=True):
            st.session_state.current_quiz = None
            st.rerun()

def main():
    """Main app function"""
    init_session_state()
    
    # Sidebar for user info
    with st.sidebar:
        st.markdown("### 👤 Player Profile")
        
        if not st.session_state.user_profile['name']:
            st.markdown("**Welcome to ManabiFun!**")
            name = st.text_input("Enter your name to start:")
            if st.button("🚀 Start Learning Adventure!", use_container_width=True):
                if name.strip():
                    st.session_state.user_profile['name'] = name.strip()
                    st.session_state.user_profile['streak'] = 1
                    st.success(f"Welcome aboard, {name}! 🎉")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please enter your name!")
        else:
            st.markdown(f"**🌟 {st.session_state.user_profile['name']}**")
            st.write(f"**Total XP:** {st.session_state.user_profile['total_xp']} 🌟")
            st.write(f"**Streak:** {st.session_state.user_profile['streak']} days 🔥")
            st.write(f"**Islands:** {len(st.session_state.user_profile['completed_islands'])}/5 🏆")
            
            # Quick stats
            if st.session_state.user_profile['completed_islands']:
                st.markdown("**Completed Islands:**")
                for island in st.session_state.user_profile['completed_islands']:
                    island_names = {
                        'grammar': '🏝️ Grammar Basics',
                        'articles': '🏖️ Articles',
                        'synonyms': '🌴 Synonyms',
                        'antonyms': '🏛️ Antonyms',
                        'sentences': '🏰 Sentences'
                    }
                    st.write(f"✅ {island_names.get(island, island.title())}")
            
            st.markdown("---")
            
            if st.button("🔄 Reset Progress", use_container_width=True):
                if st.button("⚠️ Confirm Reset", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            
            # App info
            st.markdown("---")
            st.markdown("""
            ### 📱 About ManabiFun
            - **🎯 Goal:** Master English fundamentals
            - **🏝️ Islands:** 5 learning topics
            - **🤖 AI-Powered:** Smart recommendations
            - **📊 Analytics:** Track your progress
            - **🎮 Gamified:** XP, streaks, achievements
            """)
    
    # Main content
    if not st.session_state.user_profile['name']:
        # Welcome screen
        st.markdown("""
        # 🏝️ Welcome to ManabiFun!
        ### Your AI-Powered English Learning Adventure
        
        Embark on an exciting journey through 5 magical learning islands:
        - 🏝️ **Grammar Basics Island** - Master fundamental grammar rules
        - 🏖️ **Articles Island** - Learn when to use a, an, the
        - 🌴 **Synonyms Island** - Discover words with similar meanings  
        - 🏛️ **Antonyms Island** - Explore opposite words
        - 🏰 **Sentence Formation Island** - Build perfect sentences
        
        ### ✨ Features:
        - **250 Questions** across 5 topics
        - **AI-Powered Analytics** to identify weak areas
        - **Gamified Learning** with XP and streaks
        - **Interactive Quizzes** with instant feedback
        - **Progress Tracking** with detailed charts
        
        **👈 Enter your name in the sidebar to begin your adventure!**
        """)
        
        # Demo images or animations could go here
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎯 **Personalized Learning**\n\nAI analyzes your performance and suggests areas for improvement")
        with col2:
            st.success("📊 **Detailed Analytics**\n\nTrack your progress with beautiful charts and insights")
        with col3:
            st.warning("🏆 **Achievement System**\n\nEarn XP, maintain streaks, and unlock new challenges")
            
    elif st.session_state.current_quiz is None:
        main_dashboard()
    else:
        quiz_interface()

# Run the app
if __name__ == "__main__":
    main()
