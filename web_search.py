import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
import time
from typing import List, Dict, Optional
import re

class WebSearchLLM:
    """
    A web search and content extraction tool designed for LLM consumption.
    Supports multiple search engines and extracts clean, structured content.
    """
    
    def __init__(self):
        self.session = requests.Session()
        # self.session.headers.update({
        #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        #     'Accept-Language': 'en-US,en;q=0.9'
        # })
    
    def resolve_redirect_url(self, url: str) -> str:
        """
        Resolve redirect URLs to get the actual destination URL
        """
        try:
            # Handle Bing redirect URLs
            if 'bing.com/ck/a?' in url:
                # Extract the actual URL from the redirect parameter
                parsed = urllib.parse.urlparse(url)
                query_params = urllib.parse.parse_qs(parsed.query)
                if 'u' in query_params:
                    # Bing encodes the URL in base64-like format, decode it
                    encoded_url = query_params['u'][0]
                    # Remove the 'a1' prefix and decode
                    if encoded_url.startswith('a1'):
                        encoded_url = encoded_url[2:]
                    try:
                        actual_url = urllib.parse.unquote(encoded_url)
                        return actual_url
                    except:
                        pass
            
            # For other redirect URLs, follow the redirect
            response = self.session.head(url, allow_redirects=True, timeout=10)
            return response.url
            
        except Exception as e:
            print(f"Error resolving redirect for {url}: {e}")
            return url
    
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
    
    def search_bing(self, query: str, num_results: int = 10, language: str = 'en') -> List[Dict]:
        """
        Search Bing and return structured results with resolved URLs
        """
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote_plus(query)
            
            # Force English results with language and market parameters
            url = f"https://www.bing.com/search?q={encoded_query}&count={num_results}&setlang={language}&mkt={language}-US&ensearch=1"
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse Bing search results
            for result in soup.find_all('li', class_='b_algo'):
                try:
                    title_elem = result.find('h2')
                    if not title_elem:
                        continue
                    
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    original_url = link_elem.get('href')
                    
                    # Resolve redirect URL to get actual URL
                    resolved_url = self.resolve_redirect_url(original_url)
                    
                    # Get snippet/description
                    snippet_elem = result.find('p') or result.find('div', class_='b_caption')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    results.append({
                        'title': title,
                        'url': resolved_url,
                        'original_url': original_url,  # Keep original for debugging
                        'snippet': snippet,
                        'source': 'bing'
                    })
                except Exception as e:
                    print(f"Error parsing result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Bing search error: {e}")
            return []
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Alternative search using DuckDuckGo
        """
        try:
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Add instant answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': 'duckduckgo_instant'
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:num_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'duckduckgo_related'
                    })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def extract_content(self, url: str, max_length: int = 2000) -> Dict:
        """
        Extract and clean content from a webpage for LLM consumption
        """
        try:
            # Skip if URL is still a redirect or invalid
            if not url or 'bing.com/ck/a?' in url:
                return {
                    'title': "",
                    'content': "Invalid or redirect URL",
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
                         content_extracts: List[Dict] = None) -> str:
        """
        Create a structured prompt for LLM based on search results and content
        """
        prompt = f"# Search Query: {query}\n\n"
        
        # Add search results
        prompt += "## Search Results:\n\n"
        for i, result in enumerate(search_results, 1):
            prompt += f"**Result {i}:**\n"
            prompt += f"- Title: {result['title']}\n"
            prompt += f"- URL: {result['url']}\n"
            prompt += f"- Snippet: {result['snippet']}\n"
            prompt += f"- Source: {result['source']}\n\n"
        
        # Add extracted content if available
        if content_extracts:
            prompt += "## Extracted Content:\n\n"
            for i, extract in enumerate(content_extracts, 1):
                if extract['success']:
                    prompt += f"**Content {i} from {extract['url']}:**\n"
                    prompt += f"Title: {extract['title']}\n"
                    prompt += f"Content: {extract['content']}\n\n"
                else:
                    prompt += f"**Content {i} from {extract['url']}:**\n"
                    prompt += f"Failed to extract: {extract['content']}\n\n"
        
        prompt += "## Instructions for LLM:\n"
        prompt += "Based on the search results and extracted content above, please provide a comprehensive answer to the query. "
        prompt += "Cite the sources when making specific claims and indicate if information is from search snippets vs full content extraction."
        
        return prompt
    
    def comprehensive_search(self, query: str, extract_content: bool = True, 
                           include_wikipedia: bool = True, num_results: int = 5,
                           language: str = 'en') -> Dict:
        """
        Perform comprehensive search and content extraction
        """
        results = {
            'query': query,
            'search_results': [],
            'content_extracts': [],
            'wikipedia_summary': None,
            'llm_prompt': ""
        }
        
        # Perform search
        search_results = self.search_bing(query, num_results, language)
        if not search_results:
            print("Bing search failed, trying DuckDuckGo...")
            search_results = self.search_duckduckgo(query, num_results)
        
        results['search_results'] = search_results
        
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
        results['llm_prompt'] = self.create_llm_prompt(
            query, search_results, results['content_extracts']
        )
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the search tool
    searcher = WebSearchLLM()
    
    # Example: Test redirect resolution
    print("=== Testing Redirect Resolution ===")
    test_redirect = "https://www.bing.com/ck/a?!&&p=f57492be33f656792504093de368c200ae687f49b880d75b9943392acd1d0331JmltdHM9MTc1MTg0NjQwMA&ptn=3&ver=2&hsh=4&fclid=15ab84ce-d942-6afc-1984-92ecd8566be0&u=a1aHR0cHM6Ly9pZC53aWtpcGVkaWEub3JnL3dpa2kvUHJhYm93b19TdWJpYW50bw&ntb=1"
    resolved = searcher.resolve_redirect_url(test_redirect)
    print(f"Original: {test_redirect}")
    print(f"Resolved: {resolved}")
    
    # Example: Comprehensive search for LLM
    print("\n=== Comprehensive Search for LLM ===")
    comprehensive_result = searcher.comprehensive_search(
        "donald trump", 
        extract_content=True, 
        include_wikipedia=True,
        num_results=3
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
    
    # Save results to file
    with open('search_results.json', 'w', encoding='utf-8') as f:
        # Remove the llm_prompt for JSON serialization (it's too long)
        save_data = {k: v for k, v in comprehensive_result.items() if k != 'llm_prompt'}
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print("Results saved to search_results.json")
    
    with open('llm_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(comprehensive_result['llm_prompt'])
    
    print("LLM prompt saved to llm_prompt.txt")