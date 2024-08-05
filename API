pip install google-generativeai
pip install pymupdf

import google.generativeai as genai
import os
import fitz  # PyMuPDF

# Global vars
history = []
text = ""


# Credentials
GEMINI_API_KEY = "AIzaSyDtiumsIRdLswIEww043i7UxysB-wT9-Mw"
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    global text
    """Extract text from a PDF file."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    return text



def main_menu():
    while True:
        print('1. Proceed with the chat\n2. Exit the chat')
        choice = int(input('Enter 1 or 2 > '))
        if choice == 1:
            chat()
        elif choice == 2:
            print('Thank you for visiting. Goodbye!')
            break
        else:
            print('Invalid choice. Please select 1 or 2.')
    return True


def chat():
    global text, history
    print("Thank you for reaching out to us, you are safe, we are here to hear you out.")

    if not text:
        pdf_path = '/Users/marinazub/Desktop/Ai for MH/coding/PHQ9.pdf' #update the path with the document location
        extract_text_from_pdf(pdf_path)

    genai.embed_content(model="models/text-embedding-004", content = text, task_type="document") #embedding Try to add extrac knowledge
    


    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=(
            "You are a Suicide prevention specialist with 25 years of experience. \n"
            "You are chatting with a young male 15-29 years old who's probably at risk of suicide.\n"
            "Your goal is to determine the risk level of suicide in the next week, month, and 6 months.\n\n"
            "Ask him questions using the questions from the questionnaires: PHQ-9, PHQ-2, SSQ-SR, C-SSRS, ASQ. "
            "Ask those questions in a conversational format, gently and nicely as this person is under stigma and fear of being shamed and misunderstood. "
            "Ask about his socio-demographic and cultural background, and recent events.\n\n"
            "Keep the conversation 3 minutes long at most.\n\n"
            "At the end of the conversation, analyze the answers and assess possible suicide risk. "
            "Provide suggestions on the next steps. The name of the provider is Juliana Feels. When you start the interaction, don't forget to introduce yourself as it makes the interaction more personalized."
        ), # prompt. try to tackle to get the "right" outcome.
    )

    chat_session = model.start_chat(history=history)

    # Initial user input
    initial_input = input("Tell me, what brings you here and what bothers you? ")
    response = chat_session.send_message(initial_input)
    print("Model Response:", response.text)

    # Update history
    history.append({"user": initial_input, "model": response.text})

    while True:
        user_input = input("Proceed with the chat: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for reaching out. Take care!")
            break

        # Send user input and get model response
        response = chat_session.send_message(user_input)
        print("Model Response:", response.text)

        # Update history
        history.append({"user": user_input, "model": response.text})

# Main entry point
if __name__ == "__main__":
    main_menu()
