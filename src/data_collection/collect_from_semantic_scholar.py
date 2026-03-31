"""
Production-grade script to collect inflation-related abstracts from Semantic Scholar and arXiv APIs.

This script:
- Queries Semantic Scholar API with inflation-focused keywords
- Queries arXiv API as a secondary source
- Deduplicates results by DOI and title
- Saves results to JSON with full metadata
- Includes comprehensive error handling and logging
"""

import requests
import json
import os
import sys
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Request configuration
REQUEST_TIMEOUT = 15
SEMANTIC_SCHOLAR_DELAY = 3.5  # seconds between requests (conservative for rate limiting)
ARXIV_DELAY = 2.0  # seconds between requests
SEMANTIC_SCHOLAR_LIMIT_PER_QUERY = 100  # Results per query
ARXIV_LIMIT_PER_QUERY = 50  # Results per query
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # exponential backoff multiplier

# Output
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "collected_abstracts_raw.json")

# Inflation keywords targeting economics (71% positive rate expected)
SEMANTIC_SCHOLAR_QUERIES = [
    "inflation forecasting", "CPI prediction", "inflation rate modeling",
    "monetary policy inflation", "deflation economics", "inflation expectations",
    "Phillips curve", "stagflation", "hyperinflation", "price stability central bank",
    "inflation targeting", "ARIMA inflation", "neural network inflation prediction",
    "time series inflation", "inflation dynamics", "wage price spiral",
    "demand pull inflation", "cost push inflation", "measure inflation",
    "inflation shock", "inflation persistence"
]

ARXIV_QUERIES = [
    "inflation forecasting", "CPI prediction", "monetary policy",
    "price dynamics", "inflation expectations", "deflation"
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    pass

class APIError(DataCollectionError):
    """API communication error."""
    pass

class ValidationError(DataCollectionError):
    """Data validation error."""
    pass

# ============================================================================
# SEMANTIC SCHOLAR COLLECTOR
# ============================================================================

class SemanticScholarCollector:
    """Handles data collection from Semantic Scholar API."""
    
    def __init__(self):
        self.papers: List[Dict] = []
        self.seen_dois: set = set()
        self.seen_titles: set = set()
        self.request_count: int = 0
        
    def _is_valid_abstract(self, abstract: Optional[str]) -> bool:
        """Validate abstract has minimum length and content."""
        if not abstract:
            return False
        return len(abstract.strip()) >= 100
    
    def _extract_doi(self, paper: Dict) -> Optional[str]:
        """Extract DOI from paper metadata."""
        try:
            external_ids = paper.get("externalIds", {})
            if isinstance(external_ids, dict) and "DOI" in external_ids:
                return external_ids["DOI"]
        except Exception as e:
            logger.debug(f"Error extracting DOI: {e}")
        return None
    
    def _deduplicate_check(self, doi: Optional[str], title: str) -> Tuple[bool, str]:
        """
        Check if paper is duplicate. Returns (is_duplicate, reason).
        """
        if doi and doi in self.seen_dois:
            return True, f"DOI already collected: {doi}"
        
        title_key = title.lower()[:80]
        if title_key in self.seen_titles:
            return True, f"Title already collected: {title[:50]}..."
        
        return False, ""
    
    def _make_request(self, query: str, offset: int = 0) -> Optional[Dict]:
        """Make request to Semantic Scholar API with retry logic."""
        params = {
            "query": query,
            "offset": offset,
            "limit": SEMANTIC_SCHOLAR_LIMIT_PER_QUERY,
            "fields": "paperId,title,abstract,venue,year,externalIds,authors,publicationVenue"
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Request attempt {attempt + 1}/{MAX_RETRIES} for query: {query}")
                response = requests.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "InflationDataCollector/1.0"}
                )
                response.raise_for_status()
                self.request_count += 1
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES - 1:
                    wait_time = REQUEST_TIMEOUT * (RETRY_BACKOFF ** attempt)
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise APIError(f"Timeout after {MAX_RETRIES} attempts for query: {query}")
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = REQUEST_TIMEOUT * (RETRY_BACKOFF ** attempt)
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise APIError(f"Connection error after {MAX_RETRIES} attempts: {e}")
                    
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    if attempt < MAX_RETRIES - 1:
                        wait_time = 60 * (RETRY_BACKOFF ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise APIError(f"Rate limited after {MAX_RETRIES} attempts")
                else:
                    raise APIError(f"HTTP {response.status_code}: {e}")
        
        return None
    
    def collect_from_query(self, query: str) -> int:
        """Collect papers from a single query. Returns count of new papers added."""
        logger.info(f"Starting collection for query: '{query}'")
        new_papers_count = 0
        offset = 0
        max_offset = SEMANTIC_SCHOLAR_LIMIT_PER_QUERY * 3  # Max 3 pages (300 papers)
        
        while offset < max_offset:
            try:
                data = self._make_request(query, offset)
                if not data:
                    break
                
                papers_in_response = data.get("data", [])
                if not papers_in_response:
                    logger.info(f"  No more papers for this query at offset {offset}")
                    break
                
                for paper in papers_in_response:
                    try:
                        # Validate abstract
                        abstract = paper.get("abstract")
                        if not self._is_valid_abstract(abstract):
                            continue
                        
                        title = paper.get("title", "").strip()
                        if not title:
                            continue
                        
                        # Extract DOI
                        doi = self._extract_doi(paper)
                        
                        # Check for duplicates
                        is_dup, reason = self._deduplicate_check(doi, title)
                        if is_dup:
                            logger.debug(f"  Skipped duplicate: {reason}")
                            continue
                        
                        # Record identifiers
                        if doi:
                            self.seen_dois.add(doi)
                        self.seen_titles.add(title.lower()[:80])
                        
                        # Create paper record
                        paper_record = {
                            "DOI": doi or f"SEMANTIC_SCHOLAR_{paper.get('paperId')}",
                            "Abstract": abstract.strip(),
                            "Title": title,
                            "Year": paper.get("year"),
                            "Venue": paper.get("venue"),
                            "Authors": [
                                {
                                    "name": author.get("name"),
                                    "authorId": author.get("authorId")
                                } for author in paper.get("authors", [])[:5]  # First 5 authors
                            ] if "authors" in paper else [],
                            "Source": "semantic_scholar",
                            "CollectionDate": datetime.now().isoformat(),
                            "Label": None
                        }
                        
                        self.papers.append(paper_record)
                        new_papers_count += 1
                        logger.debug(f"  ✓ Added: {title[:70]}... (DOI: {paper_record['DOI'][:30]}...)")
                        
                    except Exception as e:
                        logger.warning(f"Error processing paper: {e}")
                        continue
                
                offset += SEMANTIC_SCHOLAR_LIMIT_PER_QUERY
                time.sleep(SEMANTIC_SCHOLAR_DELAY)
                
            except APIError as e:
                logger.error(f"API error during collection: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error during collection: {e}")
                break
        
        logger.info(f"  Completed: {new_papers_count} new papers added for query '{query}'")
        return new_papers_count
    
    def collect_all(self) -> List[Dict]:
        """Collect from all Semantic Scholar queries."""
        logger.info("="*80)
        logger.info("SEMANTIC SCHOLAR COLLECTION START")
        logger.info("="*80)
        logger.info(f"Total queries: {len(SEMANTIC_SCHOLAR_QUERIES)}")
        
        total_collected = 0
        for i, query in enumerate(SEMANTIC_SCHOLAR_QUERIES):
            count = self.collect_from_query(query)
            total_collected += count
            logger.info(f"Progress: {i+1}/{len(SEMANTIC_SCHOLAR_QUERIES)} | Total so far: {total_collected}")
        
        logger.info(f"Semantic Scholar collection complete. Total: {total_collected} papers")
        return self.papers

# ============================================================================
# ARXIV COLLECTOR
# ============================================================================

class ArxivCollector:
    """Handles data collection from arXiv API."""
    
    def __init__(self):
        self.papers: List[Dict] = []
        self.seen_arxiv_ids: set = set()
        self.request_count: int = 0
    
    def _is_valid_abstract(self, abstract: Optional[str]) -> bool:
        """Validate abstract has minimum length."""
        if not abstract:
            return False
        return len(abstract.strip()) >= 100
    
    def _make_request(self, search_query: str) -> Optional[ET.Element]:
        """Make request to arXiv API with error handling."""
        try:
            logger.debug(f"arXiv API request for: {search_query[:50]}...")
            response = requests.get(
                ARXIV_API_URL,
                params={"search_query": search_query, "max_results": ARXIV_LIMIT_PER_QUERY},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "InflationDataCollector/1.0"}
            )
            response.raise_for_status()
            self.request_count += 1
            
            root = ET.fromstring(response.content)
            return root
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching from arXiv")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching from arXiv: {e}")
            return None
        except ET.ParseError as e:
            logger.error(f"Error parsing arXiv XML response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching from arXiv: {e}")
            return None
    
    def collect_from_query(self, query: str) -> int:
        """Collect papers from a single arXiv query."""
        logger.info(f"Starting arXiv collection for query: '{query}'")
        
        # arXiv search syntax: cat:economics AND all:{query}
        search_query = f"cat:econ.* AND all:{quote(query)}"
        
        try:
            root = self._make_request(search_query)
            if root is None:
                return 0
            
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            papers_found = 0
            
            for entry in root.findall("atom:entry", ns):
                try:
                    # Extract identifiers
                    arxiv_id_full = entry.find("atom:id", ns).text
                    arxiv_id = arxiv_id_full.split("/abs/")[-1] if arxiv_id_full else None
                    
                    if not arxiv_id or arxiv_id in self.seen_arxiv_ids:
                        continue
                    
                    # Extract content
                    title = entry.find("atom:title", ns).text.strip() if entry.find("atom:title", ns) is not None else ""
                    summary = entry.find("atom:summary", ns).text if entry.find("atom:summary", ns) is not None else None
                    published = entry.find("atom:published", ns).text if entry.find("atom:published", ns) is not None else ""
                    
                    # Validate abstract
                    if not self._is_valid_abstract(summary):
                        logger.debug(f"  Skipped: abstract too short for {arxiv_id}")
                        continue
                    
                    # Extract year from published date
                    year = int(published.split("-")[0]) if published else None
                    
                    # Record paper
                    self.seen_arxiv_ids.add(arxiv_id)
                    
                    paper_record = {
                        "DOI": f"ARXIV_{arxiv_id}",
                        "Abstract": summary.strip(),
                        "Title": title.strip(),
                        "Year": year,
                        "ArxivId": arxiv_id,
                        "Source": "arxiv",
                        "CollectionDate": datetime.now().isoformat(),
                        "Label": None
                    }
                    
                    self.papers.append(paper_record)
                    papers_found += 1
                    logger.debug(f"  ✓ Added: {title[:70]}... (arXiv: {arxiv_id})")
                    
                except Exception as e:
                    logger.warning(f"Error processing arXiv entry: {e}")
                    continue
            
            logger.info(f"  Completed: {papers_found} papers added from arXiv query '{query}'")
            return papers_found
            
        except Exception as e:
            logger.error(f"Error during arXiv collection: {e}")
            return 0
    
    def collect_all(self) -> List[Dict]:
        """Collect from all arXiv queries."""
        logger.info("="*80)
        logger.info("ARXIV COLLECTION START")
        logger.info("="*80)
        logger.info(f"Total queries: {len(ARXIV_QUERIES)}")
        
        total_collected = 0
        for i, query in enumerate(ARXIV_QUERIES):
            count = self.collect_from_query(query)
            total_collected += count
            logger.info(f"Progress: {i+1}/{len(ARXIV_QUERIES)} | Total so far: {total_collected}")
            time.sleep(ARXIV_DELAY)
        
        logger.info(f"arXiv collection complete. Total: {total_collected} papers")
        return self.papers

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class InflationDataCollector:
    """Orchestrates data collection from multiple sources."""
    
    def __init__(self):
        self.ss_collector = SemanticScholarCollector()
        self.arxiv_collector = ArxivCollector()
        self.all_papers: List[Dict] = []
    
    def _global_deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicates across all sources by title."""
        logger.info("Performing global deduplication...")
        
        seen_titles = set()
        deduplicated = []
        duplicates_removed = 0
        
        for paper in papers:
            title_key = paper.get("Title", "").lower()[:80]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                deduplicated.append(paper)
            else:
                duplicates_removed += 1
                logger.debug(f"  Removed duplicate: {paper.get('Title', 'Unknown')[:60]}...")
        
        logger.info(f"Deduplication complete: {duplicates_removed} duplicates removed")
        logger.info(f"Total unique papers: {len(deduplicated)}")
        return deduplicated
    
    def _validate_papers(self, papers: List[Dict]) -> None:
        """Validate collected papers before saving."""
        logger.info("Validating collected papers...")
        
        for paper in papers:
            if not paper.get("DOI"):
                raise ValidationError("Paper missing DOI")
            if not paper.get("Abstract"):
                raise ValidationError("Paper missing Abstract")
            if not paper.get("Title"):
                raise ValidationError("Paper missing Title")
            if paper.get("Label") is not None and not isinstance(paper["Label"], (int, type(None))):
                raise ValidationError(f"Invalid Label type: {type(paper['Label'])}")
        
        logger.info(f"Validation passed for {len(papers)} papers")
    
    def _save_results(self, papers: List[Dict]) -> str:
        """Save collected papers to JSON file."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"Saving {len(papers)} papers to {OUTPUT_FILE}...")
        
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            
            # Verify file was written
            if not os.path.exists(OUTPUT_FILE):
                raise IOError(f"File not created: {OUTPUT_FILE}")
            
            file_size = os.path.getsize(OUTPUT_FILE)
            logger.info(f"Successfully saved to {OUTPUT_FILE} ({file_size} bytes)")
            return OUTPUT_FILE
            
        except IOError as e:
            logger.error(f"Failed to save file: {e}")
            raise
    
    def collect_all(self) -> str:
        """Execute complete collection workflow."""
        logger.info("="*80)
        logger.info("INFLATION DATA COLLECTION PIPELINE START")
        logger.info(f"Start time: {datetime.now().isoformat()}")
        logger.info("="*80)
        
        try:
            # Phase 1: Semantic Scholar
            ss_papers = self.ss_collector.collect_all()
            logger.info(f"Semantic Scholar papers collected: {len(ss_papers)}")
            
            # Phase 2: arXiv
            arxiv_papers = self.arxiv_collector.collect_all()
            logger.info(f"arXiv papers collected: {len(arxiv_papers)}")
            
            # Phase 3: Merge
            all_papers = ss_papers + arxiv_papers
            logger.info(f"Total papers before deduplication: {len(all_papers)}")
            
            # Phase 4: Deduplicate
            self.all_papers = self._global_deduplicate(all_papers)
            
            # Phase 5: Validate
            self._validate_papers(self.all_papers)
            
            # Phase 6: Save
            output_file = self._save_results(self.all_papers)
            
            # Summary
            logger.info("="*80)
            logger.info("COLLECTION COMPLETE")
            logger.info("="*80)
            logger.info(f"Total papers collected: {len(self.all_papers)}")
            logger.info(f"Semantic Scholar: {len(ss_papers)} | arXiv: {len(arxiv_papers)}")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Total API requests: {self.ss_collector.request_count + self.arxiv_collector.request_count}")
            logger.info(f"End time: {datetime.now().isoformat()}")
            
            return output_file
            
        except Exception as e:
            logger.critical(f"Collection failed: {e}", exc_info=True)
            raise

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        collector = InflationDataCollector()
        output_file = collector.collect_all()
        print(f"\n✓ SUCCESS: Data collection complete")
        print(f"Output: {output_file}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
