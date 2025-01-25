# streamlit_app.py

import os
import glob
import json
import logging
import streamlit as st
from datetime import datetime

# News aggregator imports
from pygooglenews import GoogleNews
from bs4 import BeautifulSoup

# LLM imports
import pycountry
import ollama
from langchain.schema import AIMessage
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)

# Folder to store raw/clean data
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# LLM model
MODEL_NAME = "deepseek-r1:8b"  # can be any LLM model

# Danger keywords
DANGER_KEYWORDS = {
    "war", "violence", "bloodshed", "conflict", "collapse",
    "fall", "disease", "illness", "outbreak", "danger",
    "dangerous", "crisis", "unrest", "rebels", "extremists",
    "plague", "nanovirus", "riots", "riot", "looted", "emergency",
    "disaster", "terror", "attack", "explosion", "chaos", "shortages"
}


def fetch_and_clean_news():
    """Fetch top news, save raw + clean, keep only one file each, return path of cleaned file."""
    st.write("Fetching news from Google...")  # Streamlit feedback
    gn = GoogleNews(lang='en', country='US')
    top_stories = gn.top_news()

    raw_articles = []
    for entry in top_stories['entries']:
        raw_articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "content": entry.summary,
        })

    current_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    raw_file_name = os.path.join(DATA_FOLDER, f"news_raw_{current_time}.json")
    with open(raw_file_name, 'w', encoding="utf-8") as raw_file:
        json.dump(raw_articles, raw_file, indent=4)

    # Prepare cleaned text
    cleaned_articles = []
    for art in raw_articles:
        soup = BeautifulSoup(art["content"], "html.parser")
        cleaned_content = soup.get_text(separator=" ")
        cleaned_articles.append(
            f"Published: {art['published']}\n"
            f"Headline: {art['title']}\n"
            f"Content: {cleaned_content}\n"
        )

    clean_file_name = os.path.join(DATA_FOLDER, f"news_clean_{current_time}.txt")
    with open(clean_file_name, 'w', encoding="utf-8") as clean_file:
        clean_file.write("\n\n".join(cleaned_articles))

    # Manage file limits (only keep 1 raw + 1 clean)
    manage_file_limits(DATA_FOLDER, "news_raw_*.json", 1)
    manage_file_limits(DATA_FOLDER, "news_clean_*.txt", 1)

    return clean_file_name


def manage_file_limits(directory, pattern, max_files):
    """Keep only the newest `max_files` matches, removing older ones."""
    files = sorted(
        glob.glob(os.path.join(directory, pattern)),
        key=os.path.getmtime
    )
    while len(files) > max_files:
        oldest = files.pop(0)
        os.remove(oldest)


def build_country_set():
    """Get recognized country names (lowercase)."""
    all_countries = set()
    for c in pycountry.countries:
        all_countries.add(c.name.lower())
        off_name = getattr(c, "official_name", None)
        if off_name:
            all_countries.add(off_name.lower())
        com_name = getattr(c, "common_name", None)
        if com_name:
            all_countries.add(com_name.lower())
    return all_countries


def find_country_danger_lines(text: str, country_names: set[str]) -> dict[str, list[str]]:
    lines = text.split("\n")
    danger_map = {}
    danger_lower = {dk.lower() for dk in DANGER_KEYWORDS}

    for line in lines:
        line_lower = line.lower()
        # if it has a danger keyword
        if any(kw in line_lower for kw in danger_lower):
            # check if a recognized country is also in line
            matching = [c for c in country_names if c in line_lower]
            for lc_country in matching:
                display_country = lc_country.title()
                if display_country not in danger_map:
                    danger_map[display_country] = []
                if line not in danger_map[display_country]:
                    danger_map[display_country].append(line.strip())
    return danger_map


def short_summarize(llm: ChatOllama, line: str) -> str:
    system_msg = (
        "You summarize lines about crises or danger. "
        "Output a short, 5-word max summary. No disclaimers."
    )
    user_msg = f"Line: {line}\nShort summary:"
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ])
    if isinstance(resp, AIMessage):
        return resp.content.strip()
    return str(resp)


def analyze_danger(clean_file_path: str):
    """Analyze the cleaned file for top 3 dangerous countries."""
    if not os.path.exists(clean_file_path):
        return ["No .txt file found for LLM analysis."]

    # read text
    try:
        with open(clean_file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
    except Exception as e:
        return [f"Error reading {clean_file_path}: {e}"]

    # find dangerous countries
    all_countries = build_country_set()
    danger_map = find_country_danger_lines(text_data, all_countries)

    if not danger_map:
        return ["No dangerous info found in text."]

    # sort by # lines
    sorted_countries = sorted(danger_map.keys(), key=lambda c: len(danger_map[c]), reverse=True)
    top_countries = sorted_countries[:3]

    llm = ChatOllama(model=MODEL_NAME)
    output_lines = []
    rank = 1
    # always produce 3 bullet points; fill placeholders if needed
    for i in range(3):
        if i < len(top_countries):
            country = top_countries[i]
            line_for_country = danger_map[country][0]  # first line
            summary = short_summarize(llm, line_for_country)
            output_lines.append(f"{rank}) {country} - {summary}")
        else:
            # placeholder
            output_lines.append(f"{rank}) (No other country found)")
        rank += 1

    return output_lines


########################
# STREAMLIT UI
########################

def main():
    st.title("News Danger Web App")
    st.write("Fetch top news, identify up to 3 dangerous countries, and display results.")

    # Button to fetch & analyze
    if st.button("Fetch & Analyze Now"):
        st.write("Step 1: Fetching and cleaning news...")
        clean_file_path = fetch_and_clean_news()
        st.write(f"Cleaned file: `{clean_file_path}`")

        st.write("Step 2: Analyzing for danger...")
        results = analyze_danger(clean_file_path)
        st.write("### Top 3 Dangerous Countries:")
        for line in results:
            st.write(line)

if __name__ == "__main__":
    main()
