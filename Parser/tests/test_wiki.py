import os

import wikipediaapi as wiki

# Define the Wikipedia page and language

page_titles = [
    # LIGO Specific Wikis
    "LIGO",
    "LIGO_Scientific_Collaboration",
    "KAGRA",
    "Virgo_interferometer",
    # Gravitational Wave Science Related
    "Gravitational_wave",
    "Electromagnetic_radiation",
    # Measurement Technology Related
    "Interferometry",
    "Laser",
]

language = "en"

# Create a Wikipedia user agent (required by the API)
user_agent = "GRAVITYbot/1.0 (https://www.zooniverse.org/projects/zooniverse/gravity-spy; aosmith@syr.edu)"

# Initialize the Wikipedia API wrapper
wiki_wiki = wiki.Wikipedia(
    language=language,
    user_agent=user_agent,
    extract_format=wiki.ExtractFormat.WIKI,  # Get plain text, not HTML
)

# Fetch the page
for page_title in page_titles:
    page = wiki_wiki.page(page_title)
    # Check if the page exists
    if not page.exists():
        print(f"Error: The page '{page_title}' does not exist on Wikipedia.")
    else:
        # Get the main text content
        page_text = page.text

        # Define the output filename
        output_filename = f"{page_title.replace(' ', '_')}.txt"

        # Save the text to a file
        with open(output_filename, "w", encoding="utf-8") as text_file:
            text_file.write(page_text)

        print(
            f"Successfully saved the main text of '{page_title}' to '{output_filename}'."
        )
