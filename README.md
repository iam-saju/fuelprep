# UPSC Prep Hub (UI Branded as "fuelprep")

##  Overview

UPSC Prep Hub is an AI-powered web application designed to assist UPSC (Union Public Service Commission) aspirants in their preliminary examination preparation. It leverages generative AI models (Google Gemini and potentially Perplexity for underlying research) to dynamically create distinct study cards across various syllabus categories, providing a continuous and varied learning experience. The application also includes features for saving important topics, practicing with AI-generated questions, and analyzing study progress.

##  Key Features

* **AI-Generated Study Cards:** Dynamically generates study cards on distinct subtopics for all UPSC Prelims subjects.
    * **Continuous Scroll:** New cards are fetched as the user scrolls, providing an endless stream of information.
    * **Distinct Subtopics:** AI is prompted to provide unique subtopics to ensure variety and comprehensive coverage.
* **Category-Based Filtering:** Users can filter cards by specific UPSC syllabus categories or view cards from "All" categories for a diverse overview.
* **Card Refresh Options:**
    * Refresh content of an existing card.
    * Generate a completely new topic within the same subject.
    * Refresh related links and information for a card.
* **Personal Dashboard:**
    * Save important study cards to a personal dashboard for quick access.
    * Dashboard persists across sessions using browser LocalStorage.
* **Study Analytics:**
    * **Topics by Subject Chart:** Visualize the distribution of saved dashboard topics across different subjects.
    * **Subject Coverage Progress:** Track how many topics from the system have been added to the dashboard per subject.
    * **Dashboard Topic Mindmap:** View an interactive mindmap of topics saved in the dashboard, categorized by subject (powered by Markmap.js).
    * **PYQ Distribution by Year:** (Placeholder for future integration with actual PYQ data) Visualize PYQ distribution.
    * **General Stats:** Total cards in the system, items in the dashboard, etc.
* **AI Question Generator:**
    * Generate UPSC-style MCQs or descriptive questions on any user-provided topic using AI.
* **Related Practice Questions (MCQs):**
    * For each detailed study card, generate relevant MCQs for practice, complete with options, answers, and relevance explanations.

## 🛠️ Technologies Used

**Backend:**
* **Language:** Python 3.x
* **Framework:** FastAPI
* **AI Integration:**
    * Google Gemini API (Primary for card generation, PYQs, Q&A)
    * Perplexity API (Potentially used as an underlying research tool by Gemini or for specific deep-dive prompts, configured with `sonar-pro` model)
* **Libraries:** `uvicorn`, `python-dotenv`, `google-generativeai`, `requests`

**Frontend:**
* **Core:** HTML5, CSS3, JavaScript (ES6+)
* **Framework:** Vue.js 3 (Global Build)
* **Styling:** Tailwind CSS
* **Charting:** Chart.js
* **Mindmapping:** Markmap.js
* **Interactivity (Minor):** Alpine.js (for dropdowns, if re-enabled)

**Data Storage (Frontend):**
* Browser LocalStorage: For persisting Dashboard topics and Read Later items.

## 📂 Project Structure (Simplified)
