import os
import logging
import pycountry
import ollama
from datetime import datetime

from langchain.schema import AIMessage
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)

DATA_FOLDER = "./data"
MODEL_NAME = "llama3.2:latest"

# “Danger” keywords or phrases
DANGER_KEYWORDS = {
    "war", "violence", "bloodshed", "conflict", "collapse",
    "fall", "disease", "illness", "outbreak", "danger",
    "dangerous", "crisis", "unrest", "rebels", "extremists",
    "plague", "nanovirus", "riots", "riot", "looted", "emergency",
    "disaster", "terror", "attack", "explosion", "chaos", "shortages"
}

def get_most_recent_txt_file(folder_path: str) -> str | None:
    """Return the newest .txt file in `folder_path`, or None if none exist."""
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]
    if not txt_files:
        return None
    # Sort descending by modification time
    txt_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    return os.path.join(folder_path, txt_files[0])

def ingest_txt(txt_path: str) -> str | None:
    """Read and return the entire content of a .txt file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            data = f.read()
        logging.info(f"Loaded text from: {txt_path}")
        return data
    except Exception as e:
        logging.error(f"Error reading {txt_path}: {e}")
        return None

def build_country_set() -> set[str]:
    """
    Generate a set of all country names in lowercase using pycountry.
    This covers 200+ recognized countries. Safely handle missing attributes.
    """
    import pycountry
    all_countries = set()
    for c in pycountry.countries:
        # The 'name' is the official short name, e.g. "Finland"
        all_countries.add(c.name.lower())

        # official_name might not exist, so we use getattr safely
        off_name = getattr(c, "official_name", None)
        if off_name:
            all_countries.add(off_name.lower())

        # common_name might not exist
        com_name = getattr(c, "common_name", None)
        if com_name:
            all_countries.add(com_name.lower())

    return all_countries

def find_country_danger_lines(text: str, all_countries: set[str]) -> dict[str, list[str]]:
    """
    Returns a dict: {country -> [lines that mention that country + any danger keyword]}.
    We'll store country in Title case, e.g. "Finland".
    """
    lines = text.split("\n")
    danger_map = {}
    danger_lower = {dk.lower() for dk in DANGER_KEYWORDS}

    for line in lines:
        line_lower = line.lower()

        # Check if any danger keyword is present
        if any(kw in line_lower for kw in danger_lower):
            # Then check if any recognized country name is in that same line
            matching_countries = [c for c in all_countries if c in line_lower]
            for lc_country in matching_countries:
                display_country = lc_country.title()  # e.g. "finland" -> "Finland"
                if display_country not in danger_map:
                    danger_map[display_country] = []
                # Avoid duplicates
                if line not in danger_map[display_country]:
                    danger_map[display_country].append(line.strip())

    logging.info(f"Constructed country->danger map: {danger_map}")
    return danger_map

def short_summarize(llm: ChatOllama, line: str) -> str:
    """
    Summarize a single line with the LLM, restricting to 5 words max. No disclaimers.
    """
    system_msg = (
        "You summarize lines about crises or danger. "
        "Output a short, 5-word max summary. No disclaimers."
    )
    user_msg = f"Line: {line}\nShort summary:"

    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ])
    if isinstance(response, AIMessage):
        return response.content.strip()
    return str(response)

def main():
    # 1) Find the newest .txt file
    txt_path = get_most_recent_txt_file(DATA_FOLDER)
    if not txt_path:
        print("No .txt file found.")
        return

    # 2) Load the text
    text_data = ingest_txt(txt_path)
    if not text_data:
        print("Empty text data.")
        return

    # 3) Build the set of recognized countries
    all_countries = build_country_set()

    # 4) Identify lines referencing any recognized country + danger word
    danger_map = find_country_danger_lines(text_data, all_countries)
    if not danger_map:
        print("=== Response ===\nNo information available.")
        return

    # 5) Sort countries by how many lines mention them
    sorted_countries = sorted(danger_map.keys(), key=lambda c: len(danger_map[c]), reverse=True)

    print("=== Response ===")
    # We'll *always* print 3 bullet points. If we have fewer countries, fill in placeholders.

    llm = ChatOllama(model=MODEL_NAME)

    # For i in range(3) -> pick the i-th country if available
    for rank in range(1, 4):
        if rank-1 < len(sorted_countries):
            # we have a real country
            country = sorted_countries[rank-1]
            lines_for_country = danger_map[country]
            if lines_for_country:
                # Summarize the first line referencing that country
                summary = short_summarize(llm, lines_for_country[0])
                print(f"{rank}) {country} - {summary}")
            else:
                print(f"{rank}) No lines found for {country}??")
        else:
            # We don't have enough countries, so fill placeholder
            print(f"{rank}) (No other country found)")

if __name__ == "__main__":
    main()
