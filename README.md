# LLM-newsDanger

LLM-newsDanger is a tool that uses RAG to identify and highlight countries of concern. Using the DeepSeek-r1:8b model, it provides insights into geopolitical, social, and environmental risks.

---

## Requirements

1. **Ollama**: Ensure that Ollama is installed and running.
2. **DeepSeek-r1:8b**: The model is required for the analysis.
3. **Streamlit**: Install Streamlit to run the application.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/obyrned/LLM-newsDanger.git

2. cd LLM-newsDanger

3. pip install -r requirements.txt

4. streamlit run streamlit_app.py

5. Open your browser and go to: http://localhost:8501

Features

-Reads and processes news articles in real-time.
-Identifies countries of concern based on geopolitical, social, or environmental issues.
-Uses the advanced DeepSeek-r1:8b model for analysis.
-Customizable Model: You can modify the code to call a different model if needed, making it flexible for various use cases.

Notes

-This project relies on the DeepSeek-r1:8b model, which must be accessible during runtime.
-Ensure Ollama is running properly before starting the Streamlit application.
