import re
import asyncio
from googletrans import Translator

class URLSafeTranslator:
    translator = None
    url_pattern = None
    def __init__(self):
        self.translator = Translator()
        # Regex pattern to match various URL formats
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    def extract_urls(self, text):
        """Extract URLs from text and replace them with placeholders"""
        urls = []
        placeholders = []
        
        # Find all URLs in the text
        matches = list(self.url_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            url = match.group()
            placeholder = f"__URL_PLACEHOLDER_{i}__"
            placeholder.lower()
            urls.append(url)
            placeholders.append(placeholder)
        
        # Replace URLs with placeholders
        modified_text = text
        for i, match in enumerate(reversed(matches)):  # Reverse to maintain positions
            start, end = match.span()
            placeholderText = f"__URL_PLACEHOLDER_{len(matches)-1-i}__"
            placeholderText.lower()
            modified_text = modified_text[:start] + placeholderText + modified_text[end:]
        
        return modified_text, urls, placeholders
    
    def restore_urls(self, translated_text, urls, placeholders):
        """Restore URLs back to the translated text"""
        result = translated_text
        for i, (placeholder, url) in enumerate(zip(placeholders, urls)):
            result = result.replace(placeholder.lower(), url)
        return result
    
    async def translate(self, text, dest='en', src='auto'):
        """Translate text while preserving URLs"""
        try:
            # Extract URLs and replace with placeholders
            modified_text, urls, placeholders = self.extract_urls(text)
            
            # Translate the modified text
            if modified_text.strip():  # Only translate if there's text to translate
  
                translation = await self.translator.translate(
                    modified_text, 
                    dest=dest, 
                    src=src
                )
                translated_text = translation.text
                detected_lang = translation.src
            else:
                translated_text = modified_text
                detected_lang = src
            
            # Restore URLs
            final_result = self.restore_urls(translated_text, urls, placeholders)
            
            return {
                'original': text,
                'text': final_result,
                'translated': final_result,
                'detected_language': detected_lang,
                'target_language': dest,
                'urls_found': len(urls),
                'list_url': urls
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original': text
            }

async def main():
    # Initialize the translator
    url_safe_translator = URLSafeTranslator()
    
    print("URL-Safe Google Translator")
    print("=" * 40)
    print("Type 'quit' to exit")
    print("Format: [target_language:] your text")
    print("Example: en: Halo, bagaimana kabar Anda?")
    print("Example with URL: en: ini linknya https://youtu.be/RgKAFK5djSk?si=RFCs2iDO6CjLhUL_ ,tolong nyanyikan lagu itu ya")
    print()
    
    while True:
        try:
            user_input = input("Enter text to translate: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if user specified target language
            if ':' in user_input and len(user_input.split(':', 1)[0].strip()) <= 5:
                target_lang, text_to_translate = user_input.split(':', 1)
                target_lang = target_lang.strip()
                text_to_translate = text_to_translate.strip()
            else:
                target_lang = 'en'  # Default to English
                text_to_translate = user_input
            
            # Perform translation
            result = await url_safe_translator.translate(text_to_translate, dest=target_lang)
            print(result)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nOriginal ({result['detected_language']}): {result['original']}")
                print(f"Translated ({result['target_language']}): {result['translated']}")
                if result['urls_found'] > 0:
                    print(f"URLs preserved: {result['urls_found']}")
                print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# # Example usage
# if __name__ == "__main__":
#     # Test with the provided example
#     translator = URLSafeTranslator()
    
#     test_text = "ini linknya https://youtu.be/RgKAFK5djSk?si=RFCs2iDO6CjLhUL_ ,tolong nyanyikan lagu itu ya"
    
#     print("Testing with your example:")
#     # result = translator.translate(test_text, dest='en')
    
#     # if 'error' not in result:
#     #     print(f"Original: {result['original']}")
#     #     print(f"Translated: {result['translated']}")
#     #     print(f"URLs preserved: {result['urls_found']}")
#     # else:
#     #     print(f"Error: {result['error']}")
    
#     print("\nStarting interactive mode...")
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())