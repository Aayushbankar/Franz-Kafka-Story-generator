import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

class StoryBot:
    def __init__(self):
        self.model = "gpt-3.5-turbo"  # You can change this to gpt-4 if you have access
        self.conversation_history = []

    def generate_story_continuation(self, prompt, max_tokens=500):
        """
        Generate a continuation of the story based on the given prompt
        """
        try:
            # Add user's prompt to conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            
            # Create the system message to guide the AI
            system_message = {
                "role": "system",
                "content": "You are a creative storyteller. Continue the story in an engaging and coherent way."
            }
            
            # Generate the continuation
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[system_message] + self.conversation_history,
                max_tokens=max_tokens,
                temperature=0.7,  # Controls randomness (0.0 - 1.0)
                top_p=0.9,
                presence_penalty=0.6,
                frequency_penalty=0.6
            )
            
            # Get the generated continuation
            continuation = response.choices[0].message['content']
            
            # Add AI's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": continuation})
            
            return continuation
            
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Create instance of StoryBot
    story_bot = StoryBot()
    
    print("Welcome to StoryBot! Enter your story prompt, and I'll continue it.")
    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYour prompt: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        continuation = story_bot.generate_story_continuation(user_input)
        print("\nStory continuation:")
        print(continuation)

if __name__ == "__main__":
    main()
