import os
import feedparser
import requests
from datetime import datetime
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self):
        # List of RSS feeds from reputable news sources
        self.news_sources = {
            'Reuters': 'https://www.reutersagency.com/feed/?best-topics=tech&post_type=best',
            'Associated Press': 'https://feeds.feedburner.com/apnews/world',
            'NPR': 'https://feeds.npr.org/1001/rss.xml',
            'BBC': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'The Guardian': 'https://www.theguardian.com/world/rss',
            'Al Jazeera': 'https://www.aljazeera.com/xml/rss/all.xml'
        }
        self.clear_cache()
        self.cache_duration = 48 * 60 * 60 # Cache duration in seconds (48 hours)
        
    def clear_cache(self):
        self.cache = {}
        self.cache_time = {}
        
    def _extract_article_content(self, url: str) -> str:
        """Extract the main article content from a URL."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
                
            # Find the main article content (this is a simple implementation)
            article = soup.find('article') or soup.find(class_=['article', 'story', 'post'])
            if article:
                paragraphs = article.find_all('p')
                content = ' '.join(p.get_text().strip() for p in paragraphs)
            else:
                # Fallback to all paragraphs if no article tag found
                paragraphs = soup.find_all('p')
                content = ' '.join(p.get_text().strip() for p in paragraphs[:6])  # Reduced from 10 to 6 paragraphs
                
            # Remove "Continue reading..." text
            content = content.replace('Continue reading...', '')
            
            # Truncate to 60% of original length (3000 chars) and ensure we don't cut words
            content = content[:3000]
            last_space = content.rfind(' ')
            if last_space > 0:
                content = content[:last_space]
            return content.strip()
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return ""

    def _fetch_from_source(self, source_name: str, source_url: str) -> List[Dict]:
        """Fetch news from a single source."""
        try:
            feed = feedparser.parse(source_url)
            stories = []
            
            for entry in feed.entries:
                # Get the publication date
                if hasattr(entry, 'published_parsed'):
                    date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed'):
                    date = datetime(*entry.updated_parsed[:6])
                else:
                    date = datetime.now()
                
                # Get content from either summary or full content
                content = entry.get('summary', '')
                if not content or len(content) < 100 or len(content) > 2500:  # If summary is too short
                    content = self._extract_article_content(entry.link)
                
                story = {
                    'title': entry.title,
                    'content': content,
                    'source': source_name,
                    'url': entry.link,
                    'published_date': date,
                    'original_entity': '',  # Will be populated during modification
                    'modified_entity': ''   # Will be populated during modification
                }
                stories.append(story)
                
            return stories
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {str(e)}")
            return []

    def fetch_top_stories(self, limit: int = 10) -> List[Dict]:
        """Fetch top stories from various sources."""
        logger.info("Starting to fetch top stories...")
        current_time = time.time()
        
        # Check if we have cached stories
        if 'top_stories' in self.cache and current_time - self.cache_time.get('top_stories', 0) < self.cache_duration:
            logger.info("Returning cached stories")
            return self.cache['top_stories'][:limit]
        
        # Check if we have valid cached results
        if 'top_stories' in self.cache and current_time - self.cache_time.get('top_stories', 0) < self.cache_duration.total_seconds():
            logger.info("Returning cached stories")
            return self.cache['top_stories'][:limit]
            
        all_stories = []
        
        # Use ThreadPoolExecutor to fetch from multiple sources concurrently
        with ThreadPoolExecutor(max_workers=len(self.news_sources)) as executor:
            future_to_source = {
                executor.submit(self._fetch_from_source, name, url): name 
                for name, url in self.news_sources.items()
            }
            
            for future in future_to_source:
                stories = future.result()
                # Filter out stories without content
                stories = [story for story in stories if story['content'] and len(story['content'].strip()) > 100]
                all_stories.extend(stories)
        
        # Then deduplicate in a single pass
        seen_titles = set()
        seen_urls = set()
        seen_contents = set()
        unique_stories = []
        
        for story in all_stories:
            # Normalize title and content for comparison
            normalized_title = ''.join(c.lower() for c in story['title'] if c.isalnum())
            normalized_content = ''.join(c.lower() for c in story['content'][:200] if c.isalnum())
            url = story['url']
            
            # Check if this is a duplicate story
            if normalized_title not in seen_titles and \
               url not in seen_urls and \
               normalized_content not in seen_contents:
                seen_titles.add(normalized_title)
                seen_urls.add(url)
                seen_contents.add(normalized_content)
                unique_stories.append(story)
                logger.info(f"Added unique story: {story['title'][:50]}...")
        
        # Sort stories by date in descending order
        unique_stories.sort(key=lambda x: x['published_date'], reverse=True)

        logger.info(f"Fetched {len(unique_stories)} total stories after removing duplicates")
        logger.info(f"Returning {limit} stories after limit")
        
        # Cache the stories
        self.cache['top_stories'] = unique_stories[:limit]
        self.cache_time['top_stories'] = time.time()
        
        return unique_stories[:limit]
        
    def fetch_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """Fetch stories related to a specific topic."""
        # First get all stories
        all_stories = self.fetch_top_stories(limit=20)  # Get more stories to filter from
        
        # Filter stories by topic
        topic = topic.lower()
        filtered_stories = [
            story for story in all_stories
            if topic in story['title'].lower() or topic in story['content'].lower()
        ]
        
        return filtered_stories[:limit] 