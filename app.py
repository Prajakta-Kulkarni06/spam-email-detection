import os
import joblib
import streamlit as st

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="MailGuard AI | Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "spam_model.joblib"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
@import url(
 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

.stApp {
    background: linear-gradient(135deg, #f5f8ff, #edf2ff);
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #101b35;
}

[data-testid="stSidebar"] * {
    color: #f4f7ff;
}

.hero {
    background: linear-gradient(120deg, #101b35, #234c9b);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(25, 52, 110, .15);
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dce7ff;
    font-size: 16px;
}

.panel {
    background: white;
    border: 1px solid #e1e8f5;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 5px 20px rgba(32, 56, 100, .04);
}

.small-muted {
    color: #63718a;
    font-size: 14px;
}

div.stButton > button {
    background: #315efb;
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 48px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #2348d8;
    color: white;
}

.footer {
    text-align: center;
    color: #71809a;
    font-size: 12px;
    padding: 25px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception:
    st.error(
        "The model could not be loaded. "
        "Make sure spam_model.joblib is in the project folder."
    )
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛡️ MailGuard AI")
    st.caption("INTELLIGENT EMAIL SECURITY")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Email Analyzer", "About Project", "Model Information"]
    )

    st.divider()
    st.markdown("### System status")
    st.success("Machine learning model loaded")
    st.caption("Algorithm: Multinomial Naive Bayes")
    st.caption("Features: TF-IDF text analysis")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛡️ Smart Spam Detection</h1>
    <p>
        Analyze suspicious messages using machine learning.
        Identify potential spam in seconds with a clean,
        simple and intelligent interface.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- EMAIL ANALYZER ----------------
if page == "Email Analyzer":

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.markdown("### ✉️ Email content analyzer")
        st.write(
            "Paste the message you want to check below."
        )

        email_text = st.text_area(
            "Message content",
            height=250,
            placeholder=(
                "Example: Congratulations! "
                "You have won a prize. Click here to claim..."
            ),
            help="Enter the text of the email or message."
        )

        analyze = st.button(
            "🔍 Analyze Message",
            use_container_width=True
        )

        if analyze:
            if not email_text.strip():
                st.warning("Please enter a message first.")
            else:
                prediction = int(model.predict([email_text])[0])

                probabilities = model.predict_proba(
                    [email_text]
                )[0]

                # Column 1 is class 0, column 2 is class 1
                spam_index = list(
                    model.named_steps["classifier"].classes_
                ).index(1)

                spam_score = float(
                    probabilities[spam_index]
                ) * 100

                st.session_state["result"] = {
                    "prediction": prediction,
                    "spam_score": spam_score
                }

    with right:
        st.markdown("### 📊 Analysis result")

        if "result" not in st.session_state:
            st.markdown("""
            <div class="panel">
                <h3>Ready to analyze</h3>
                <p class="small-muted">
                    Your prediction will appear here
                    after you submit a message.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            result = st.session_state["result"]
            score = result["spam_score"]

            if result["prediction"] == 1:
                st.error("🚨 Classification: SPAM")
                st.write(
                    "The model classified this message "
                    "as spam."
                )
            else:
                st.success("✅ Classification: LEGITIMATE")
                st.write(
                    "The model classified this message "
                    "as legitimate (ham)."
                )

            st.metric("Model spam score", f"{score:.1f}%")
            st.progress(int(round(score)))

            st.caption(
                "This score is the model's estimated "
                "spam probability, not a guarantee."
            )

    st.divider()

    st.markdown("### 💡 How to use this tool")

    c1, c2, c3 = st.columns(3)

    c1.info("**01 — Paste**\n\nEnter your message.")
    c2.info("**02 — Analyze**\n\nRun the classifier.")
    c3.info("**03 — Review**\n\nCheck the predicted result.")

    st.warning(
        "Educational project only. This model was trained "
        "on labeled SMS messages, so email performance may "
        "differ. Do not rely on it as a security guarantee."
    )


# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.markdown("### 📘 Project overview")

    st.markdown("""
    <div class="panel">
        <h3>Spam Email Detection Using Machine Learning</h3>
        <p>
        This project demonstrates automated text
        classification. It processes message text, extracts
        TF-IDF features, and uses a Naive Bayes classifier
        to predict whether a message is spam or legitimate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Objectives")

    st.write("- Detect suspicious or unwanted messages.")
    st.write("- Demonstrate text preprocessing and NLP features.")
    st.write("- Train and evaluate a supervised ML model.")
    st.write("- Present predictions through a web interface.")

    st.markdown("#### Machine learning workflow")

    st.code(
        "Dataset → TF-IDF → Naive Bayes → Prediction → UI",
        language="text"
    )

    st.markdown("#### Limitations")

    st.write(
        "The training data consists of SMS messages. "
        "A larger, properly labeled email dataset would "
        "be needed to evaluate real email performance."
    )


# ---------------- MODEL INFORMATION ----------------
elif page == "Model Information":

    st.markdown("### ⚙️ Model information")

    st.markdown("""
    <div class="panel">
        <h4>TF-IDF Vectorization</h4>
        <p>
        Represents text using word and phrase importance
        scores rather than raw strings.
        </p>
        <h4>Multinomial Naive Bayes</h4>
        <p>
        A probabilistic classification algorithm often
        used for text categorization.
        </p>
        <h4>Evaluation</h4>
        <p>
        The model is evaluated using a held-out test set.
        Accuracy alone does not describe all classification
        errors, so precision, recall and F1-score are also
        measured in the training notebook.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Input:** Text message")
    st.write("**Output:** Spam or legitimate")
    st.write("**Feature extraction:** TF-IDF")
    st.write("**Classifier:** Multinomial Naive Bayes")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    MAILGUARD AI · MACHINE LEARNING MINI PROJECT<br>
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)
