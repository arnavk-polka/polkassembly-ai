import os
import signal
from contextlib import contextmanager
from dotenv import load_dotenv
from google import genai

@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class GeminiClientError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, model_name=None, timeout=None):
        load_dotenv()
        
        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please check your .env file.")
        
        self.timeout = timeout if timeout is not None else float(os.getenv('API_TIMEOUT', '10'))
        self.model_name = model_name
        
        print(f"Initializing Gemini client with model: {model_name}")
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            print("✅ Client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
        
    def get_response(self, prompt, timeout=None, **kwargs):
        request_timeout = timeout or self.timeout
        
        try:
            print(f"🤖 Sending request to {self.model_name}...")
            print(f"⏱️  Timeout set to: {request_timeout} seconds")
            print(f"📝 Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            with timeout_context(request_timeout):
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    **kwargs
                )
                
                print("✅ Response received successfully")
                return response.text
                
        except TimeoutError as e:
            error_msg = f"Request timed out after {request_timeout} seconds. Try increasing timeout or check your internet connection."
            print(f"⏰ {error_msg}")
            raise TimeoutError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            
            if "api_key" in str(e).lower():
                error_msg += "\n💡 Check if your GEMINI_API_KEY is valid and has proper permissions."
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                error_msg += "\n💡 You may have exceeded your API quota or rate limits."
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                error_msg += "\n💡 Check your internet connection and try again."
            
            raise GeminiClientError(error_msg) from e

    def chat(self, messages, timeout=None, **kwargs):
        contents = []
        for msg in messages:
            contents.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        
        conversation = "\n".join(contents)
        print(f"💬 Starting chat conversation with {len(messages)} messages")
        
        return self.get_response(conversation, timeout=timeout, **kwargs)

    def test_connection(self):
        print("🔍 Testing connection...")
        try:
            test_response = self.get_response("Hello", timeout=10)
            if test_response:
                print("✅ Connection test successful!")
                return True
            print("❌ Connection test failed: empty response")
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False

if __name__ == "__main__":
    client = GeminiClient(timeout=15)
    
    print("Example 1: Basic Response")
    prompt = "What is federated learning in one sentence?"
    response = client.get_response(prompt, timeout=10)
    print(f"Response: {response}")
    print("\n" + "=" * 30 + "\n")
       