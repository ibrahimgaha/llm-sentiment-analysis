
import streamlit as st
from sentiment import analyze_sentiment

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 AI Sentiment Analyzer")
st.write(
    "Analyze customer reviews, comments, or any text "
    "using LangChain and NVIDIA AI."
)

st.divider()

text = st.text_area(
    "Enter text to analyze",
    placeholder="Example: I absolutely love this product!",
    height=160,
)

analyze = st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True,
)

if analyze:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                result = analyze_sentiment(text)

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                explanation = result["explanation"]

                st.subheader("Analysis Result")

                if sentiment == "Positive":
                    st.success(f"😊 Positive Sentiment")
                elif sentiment == "Negative":
                    st.error(f"😞 Negative Sentiment")
                else:
                    st.info(f"😐 Neutral Sentiment")

                st.metric(
                    "AI Confidence Estimate",
                    f"{confidence * 100:.1f}%",
                )

                st.progress(
                    min(max(float(confidence), 0.0), 1.0)
                )

                st.write("**Explanation:**")
                st.write(explanation)

            except Exception as error:
                st.error(f"Analysis failed: {error}")

st.divider()
st.caption("Powered by LangChain and NVIDIA AI")
