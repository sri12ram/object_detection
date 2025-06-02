import openai
import os

# Set up Azure OpenAI credentials
class OpenAIConfig:
    api_type = "azure"
    api_base = "https://object-detection-testing.openai.azure.com/"
    api_version = "2023-06-01-preview"
    api_key = os.getenv("OPENAI_API_KEY", "default_key")

    @classmethod
    def configure_openai(cls):
        try:
            openai.api_type = cls.api_type
            openai.api_base = cls.api_base
            openai.api_version = cls.api_version
            openai.api_key = cls.api_key
        except Exception as e:
            print(f"Error configuring OpenAI: {e}")

# Call this at the start of your app to configure OpenAI
OpenAIConfig.configure_openai()
