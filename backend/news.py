import random
import requests
import os 
import json
from dotenv import load_dotenv

load_dotenv()
# 1. Define the expected 2025 weightage ranges
weightage_2025 = {
    "Polity": (18, 20),
    "History": (16, 18),
    "Geography": (15, 17),
    "Economy": (13, 15),
    "Current Affairs": (22, 24),
    "Environment": (13, 15),
    "Science": (7, 9)
}

# 2. Calculate average weightage for each subject
avg_weightage_2025 = {k: (v[0] + v[1]) / 2 for k, v in weightage_2025.items()}

# 3. Prompts for each subject
prompts = {
    "Polity": "List top UPSC Prelims 2025 Polity topics (weightage 18–20%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on constitutional amendments, Supreme Court judgments, and government policies. Exclude editorials.",
    "History": "List top UPSC Prelims 2025 History topics (weightage 16–18%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on freedom struggle, cultural heritage, and archaeological discoveries. Exclude editorials.",
    "Geography": "List top UPSC Prelims 2025 Geography topics (weightage 15–17%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on disaster management, climate events, and Indian geography. Exclude editorials.",
    "Economy": "List top UPSC Prelims 2025 Economy topics (weightage 13–15%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on budget, economic policies, and major government schemes. Exclude editorials.",
    "Current Affairs": "List top UPSC Prelims 2025 Current Affairs topics (weightage 22–24%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on cabinet decisions, international summits, and major government announcements. Exclude editorials.",
    "Environment": "List top UPSC Prelims 2025 Environment topics (weightage 13–15%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on climate change, biodiversity, and environmental policies. Exclude editorials.",
    "Science": "List top UPSC Prelims 2025 Science topics (weightage 7–9%) with 2-sentence summaries and 2 recent news links (<4 months) each from The Hindu, Times of India, and Indian Express. Focus on space missions, scientific breakthroughs, and technology initiatives. Exclude editorials."
}

# 4. Weighted random selection function
def weighted_random_subject(weightage_dict):
    subjects = list(weightage_dict.keys())
    weights = list(weightage_dict.values())
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return random.choices(subjects, probabilities)[0]

# 5. PPLX API call function with improved error handling
def call_pplx_api(prompt, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",  # Updated to a more current model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

# 6. Function to safely extract content from API response
def extract_content(result):
    if result is None:
        return "Error: Failed to get response from API"
    
    # Print the full response for debugging
    print("Full API Response:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Try different possible response structures
    if 'choices' in result and len(result['choices']) > 0:
        if 'message' in result['choices'][0]:
            return result['choices'][0]['message'].get('content', 'No content found')
        elif 'text' in result['choices'][0]:
            return result['choices'][0]['text']
    
    # Alternative structure that some APIs use
    if 'content' in result:
        return result['content']
    
    if 'response' in result:
        return result['response']
    
    if 'answer' in result:
        return result['answer']
    
    # If error in response
    if 'error' in result:
        return f"API Error: {result['error']}"
    
    return f"Unexpected response structure. Available keys: {list(result.keys())}"

# 7. Main workflow
def main():
    # Check for API key
    PPLX_API_KEY =os.getenv("PPLX_API_KEY")
    if not PPLX_API_KEY:
        print("Error: PPLX_API_KEY environment variable not set!")
        print("Please set it using: export PPLX_API_KEY='your_api_key_here'")
        return
    
    # Weighted random subject selection
    subject = weighted_random_subject(avg_weightage_2025)
    print(f"Selected subject (weighted): {subject}")
    print(f"Expected weightage: {weightage_2025[subject][0]}%-{weightage_2025[subject][1]}%")

    # Get the optimized prompt for that subject
    prompt = prompts[subject]
    print(f"\nPrompt to use:")
    print(prompt)

    # Call the pplx API
    print(f"\nFetching results from Perplexity API...\n")
    result = call_pplx_api(prompt, PPLX_API_KEY)
    
    # Extract and display content
    content = extract_content(result)
    print("EXTRACTED CONTENT:")
    print("="*50)
    print(content)

if __name__ == "__main__":
    main()