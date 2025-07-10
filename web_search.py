import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
import time
from typing import List, Dict, Optional
import re
from ddgs import DDGS

class WebSearchLLM:
    """
    A web search and content extraction tool designed for LLM consumption.
    Uses DuckDuckGo via the ddgs library and extracts clean, structured content.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.ddgs = DDGS()
    
    def clean_unicode_content(self, text: str) -> str:
        """
        Clean problematic Unicode characters that cause encoding issues
        """
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
        
        # Remove other problematic characters
        text = re.sub(r'[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]', '', text)
        
        # Replace common problematic characters with ASCII equivalents
        replacements = {
            '\u2013': '-',  # en dash
            '\u2014': '--', # em dash
            '\u2018': "'",  # left single quotation mark
            '\u2019': "'",  # right single quotation mark
            '\u201c': '"',  # left double quotation mark
            '\u201d': '"',  # right double quotation mark
            '\u2026': '...', # horizontal ellipsis
            '\u00a0': ' ',  # non-breaking space
        }
        
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def search_duckduckgo(self, query: str, num_results: int = 10, region: str = 'us-en') -> List[Dict]:
        """
        Search DuckDuckGo using the ddgs library
        """
        try:
            results = []
            
            # Use the text search method from ddgs
            search_results = self.ddgs.text(
                keywords=query,
                region=region,
                safesearch='moderate',
                timelimit=None,
                max_results=num_results
            )
            
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def search_duckduckgo_news(self, query: str, num_results: int = 10, region: str = 'us-en') -> List[Dict]:
        """
        Search DuckDuckGo news using the ddgs library
        """
        try:
            results = []
            
            # Use the news search method from ddgs
            news_results = self.ddgs.news(
                keywords=query,
                region=region,
                safesearch='moderate',
                timelimit='m',  # Last month
                max_results=num_results
            )
            
            for result in news_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('body', ''),
                    'date': result.get('date', ''),
                    'source': 'duckduckgo_news'
                })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo news search error: {e}")
            return []
    
    def extract_content(self, url: str, max_length: int = 2000) -> Dict:
        """
        Extract and clean content from a webpage for LLM consumption
        """
        try:
            # Skip if URL is invalid
            if not url or not url.startswith(('http://', 'https://')):
                return {
                    'title': "",
                    'content': "Invalid URL",
                    'url': url,
                    'length': 0,
                    'success': False
                }
            
            response = self.session.get(url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title = title.get_text(strip=True) if title else ""
            
            # Extract main content
            content = ""
            
            # Try to find main content areas
            main_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.article-content', '.post-content',
                '.entry-content', '.post-body', '.article-body'
            ]
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body content
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean and truncate content
            content = re.sub(r'\s+', ' ', content)
            # Remove problematic Unicode characters
            content = self.clean_unicode_content(content)
            content = content[:max_length] + "..." if len(content) > max_length else content
            
            # Check if content is meaningful
            if len(content.strip()) < 50:
                return {
                    'title': title,
                    'content': "Content too short or failed to extract meaningful content",
                    'url': url,
                    'length': len(content),
                    'success': False
                }
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'length': len(content),
                'success': True
            }
            
        except Exception as e:
            return {
                'title': "",
                'content': f"Error extracting content: {str(e)}",
                'url': url,
                'length': 0,
                'success': False
            }
    
    def extract_wikipedia_summary(self, topic: str) -> Dict:
        """
        Extract Wikipedia summary for a topic
        """
        try:
            # Use Wikipedia API
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(topic)
            
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            return {
                'title': data.get('title', ''),
                'summary': data.get('extract', ''),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'thumbnail': data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else '',
                'success': True
            }
            
        except Exception as e:
            return {
                'title': topic,
                'summary': f"Error fetching Wikipedia summary: {str(e)}",
                'url': '',
                'thumbnail': '',
                'success': False
            }
    
    def create_llm_prompt(self, query: str, search_results: List[Dict], 
                         content_extracts: List[Dict] = None, 
                         instant_answer: Dict = None) -> str:
        """
        Create a structured prompt for LLM based on search results and content
        """
        prompt = f"# Search Query: {query}\n\n"
        
        # Add instant answer if available
        if instant_answer and instant_answer.get('success'):
            prompt += "## Instant Answer:\n\n"
            prompt += f"**{instant_answer['title']}**\n"
            prompt += f"{instant_answer['snippet']}\n\n"
        
        # Add search results
        prompt += "## Search Results:\n\n"
        for i, result in enumerate(search_results, 1):
            prompt += f"**Result {i}:**\n"
            prompt += f"- Title: {result['title']}\n"
            prompt += f"- Snippet: {result['snippet']}\n"
            if result.get('date'):
                prompt += f"- Date: {result['date']}\n"
            prompt += f"- Source: {result['source']}\n\n"
        
        # Add extracted content if available
        if content_extracts:
            prompt += "## Extracted Content:\n\n"
            for i, extract in enumerate(content_extracts, 1):
                if extract['success']:
                    prompt += f"**Content {i}:**\n"
                    prompt += f"Title: {extract['title']}\n"
                    prompt += f"Content: {extract['content']}\n\n"
        
        return prompt
    
    def comprehensive_search(self, query: str, extract_content: bool = True, 
                           include_wikipedia: bool = True, include_news: bool = False,
                           num_results: int = 5, region: str = 'us-en') -> Dict:
        """
        Perform comprehensive search and content extraction
        """
        results = {
            'query': query,
            'search_results': [],
            'news_results': [],
            'content_extracts': [],
            'wikipedia_summary': None,
            'instant_answer': None,
            'llm_prompt': ""
        }
        
        # Perform regular search
        print("Performing web search...")
        search_results = self.search_duckduckgo(query, num_results, region)
        results['search_results'] = search_results
        
        # Perform news search if requested
        if include_news:
            print("Performing news search...")
            news_results = self.search_duckduckgo_news(query, num_results, region)
            results['news_results'] = news_results
        
        # Extract content from top results
        if extract_content and search_results:
            print("Extracting content from search results...")
            for i, result in enumerate(search_results[:3]):  # Extract from top 3 results
                if result['url']:
                    print(f"Extracting content from: {result['url']}")
                    content = self.extract_content(result['url'])
                    results['content_extracts'].append(content)
                    time.sleep(1)  # Be respectful to servers
        
        # Get Wikipedia summary
        if include_wikipedia:
            print("Fetching Wikipedia summary...")
            wiki_summary = self.extract_wikipedia_summary(query)
            results['wikipedia_summary'] = wiki_summary
        
        # Create LLM prompt
        all_results = search_results + results['news_results']
        results['llm_prompt'] = self.create_llm_prompt(
            query, all_results, results['content_extracts']
        )
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the search tool
    searcher = WebSearchLLM()
    
    # Example: Comprehensive search for LLM
    print("=== Comprehensive Search for LLM ===")
    comprehensive_result = searcher.comprehensive_search(
        "python", 
        extract_content=True, 
        include_wikipedia=True,
        include_news=False,
        num_results=5
    )
    
    print("\nSearch Results:")
    for i, result in enumerate(comprehensive_result['search_results'], 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()
    
    print("\nContent Extraction Results:")
    for i, extract in enumerate(comprehensive_result['content_extracts'], 1):
        print(f"{i}. Success: {extract['success']}")
        print(f"   Title: {extract['title']}")
        print(f"   Content Length: {extract['length']}")
        print(f"   URL: {extract['url']}")
        if extract['success']:
            print(f"   Content Preview: {extract['content'][:200]}...")
        else:
            print(f"   Error: {extract['content']}")
        print()
    
    # Wikipedia summary
    if comprehensive_result['wikipedia_summary'] and comprehensive_result['wikipedia_summary']['success']:
        print("\nWikipedia Summary:")
        print(f"Title: {comprehensive_result['wikipedia_summary']['title']}")
        print(f"Summary: {comprehensive_result['wikipedia_summary']['summary'][:300]}...")
        print()
    
    # Save results to file
    with open('search_results.json', 'w', encoding='utf-8') as f:
        # Remove the llm_prompt for JSON serialization (it's too long)
        save_data = {k: v for k, v in comprehensive_result.items() if k != 'llm_prompt'}
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print("Results saved to search_results.json")
    
    with open('llm_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(comprehensive_result['llm_prompt'])
    
    print("LLM prompt saved to llm_prompt.txt")