"""
Production-grade script to auto-label inflation abstracts with manual review capability.

This script:
- Auto-labels abstracts using rule-based heuristics (relevance scoring)
- Flags low-confidence papers for manual review
- Provides interactive CLI for manual labeling
- Saves labeled results with confidence scores and audit trail
"""

import json
import os
import sys
import logging
import re
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/raw/collected_abstracts_raw.json"
OUTPUT_FILE = "data/raw/collected_abstracts_labeled.json"
STATS_FILE = "data/raw/labeling_statistics.json"
MANUAL_REVIEW_LOG = "data/raw/manual_review_log.txt"

# Confidence thresholds
AUTO_LABEL_CONFIDENCE_THRESHOLD = 0.60  # Papers with score >= this are auto-labeled
MANUAL_REVIEW_CONFIDENCE_THRESHOLD = 0.40  # Papers between this and threshold need manual review

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('labeling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class LabelingError(Exception):
    """Base exception for labeling errors."""
    pass

class InputValidationError(LabelingError):
    """Input validation error."""
    pass

# ============================================================================
# LABELING ENGINE
# ============================================================================

class InflationAbstractLabeler:
    """Rule-based labeler for inflation abstracts."""
    
    def __init__(self):
        """Initialize labeler with inflation terminology."""
        
        # Core inflation economics terms (primary indicators)
        self.core_terms = {
            "inflation rate", "cpi", "consumer price index", "deflation",
            "disinflation", "price stability", "monetary policy", "central bank",
            "inflation targeting", "price level", "price index", "gdp deflator",
            "pce", "personal consumption expenditures"
        }
        
        # Economic modeling terms (secondary indicators)
        self.model_terms = {
            "forecast", "forecasting", "predict", "prediction", "model", "modeling",
            "arima", "autoregressive", "var", "vector autoregressive", "time series",
            "neural network", "lstm", "gru", "rnn", "machine learning", "deep learning",
            "regression", "estimation", "econometric", "estimate", "estimated",
            "analysis", "analyze", "examine", "evaluate"
        }
        
        # Phillips curve and advanced inflation theory
        self.phillips_terms = {
            "phillips curve", "stagflation", "hyperinflation", "wage price",
            "expectations", "expectation formation", "inflation spiral",
            "output gap", "nairu", "natural rate", "impulse response",
            "demand pull", "cost push"
        }
        
        # Non-economic uses of "inflation" (exclusion phrases)
        self.exclude_phrases = {
            "credential inflation", "grade inflation", "resume inflation",
            "scope inflation", "feature inflation", "project inflation",
            "housing inflation", "asset inflation", "real estate bubble",
            "stock bubble", "qualitative inflation", "scope creep"
        }
        
        # False positive keywords (papers mentioning inflation but non-economically)
        self.false_positive_patterns = [
            r"grade\s+inflation",
            r"credential\s+inflation",
            r"resume\s+inflation",
            r"university\s+inflation",
            r"college\s+inflation",
        ]
        
        self.labeled_count = 0
        self.manual_review_count = 0
        self.auto_label_count = 0
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Remove extra whitespace and convert to lowercase
        text = " ".join(text.split()).lower()
        return text
    
    def _check_false_positives(self, text: str) -> bool:
        """Check if text matches false positive patterns."""
        for pattern in self.false_positive_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _count_term_matches(self, text: str, terms: set) -> int:
        """Count term matches in text."""
        count = 0
        for term in terms:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                count += 1
        return count
    
    def _calculate_confidence_score(self, text: str) -> Tuple[float, Dict]:
        """
        Calculate relevance confidence score (0.0 to 1.0).
        Returns (score, details_dict).
        """
        details = {
            "core_matches": 0,
            "model_matches": 0,
            "phillips_matches": 0,
            "false_positive": False
        }
        
        # Check false positives
        if self._check_false_positives(text):
            details["false_positive"] = True
            return 0.0, details
        
        # Count term matches
        details["core_matches"] = self._count_term_matches(text, self.core_terms)
        details["model_matches"] = self._count_term_matches(text, self.model_terms)
        details["phillips_matches"] = self._count_term_matches(text, self.phillips_terms)
        
        # Scoring algorithm
        score = 0.0
        
        # Core terms: 0.4 max contribution
        if details["core_matches"] > 0:
            score += min(0.40, details["core_matches"] * 0.15)
        
        # Model terms: 0.3 max contribution
        if details["model_matches"] > 0:
            score += min(0.30, details["model_matches"] * 0.08)
        
        # Phillips/advanced terms: 0.3 max contribution
        if details["phillips_matches"] > 0:
            score += min(0.30, details["phillips_matches"] * 0.10)
        
        return min(1.0, score), details
    
    def label_abstract(self, abstract: str, title: str = "") -> Tuple[Optional[int], float, Dict]:
        """
        Label abstract as 1 (relevant) or 0 (not relevant).
        
        Returns:
            (label, confidence_score, details_dict)
            - label: 1=relevant, 0=not relevant, None=needs manual review
            - confidence_score: 0.0 to 1.0
            - details_dict: scoring breakdown
        """
        if not abstract or not isinstance(abstract, str):
            logger.warning("Invalid abstract provided")
            return 0, 0.0, {"error": "invalid_abstract"}
        
        # Combine abstract and title for analysis
        combined_text = self._preprocess_text(abstract + " " + title)
        
        # Calculate confidence
        confidence, details = self._calculate_confidence_score(combined_text)
        
        # Decision logic
        if confidence >= AUTO_LABEL_CONFIDENCE_THRESHOLD:
            # High confidence: auto-label
            label = 1 if confidence >= 0.50 else 0
            return label, confidence, details
        else:
            # Low confidence: flag for manual review
            return None, confidence, details
    
    def label_batch(self, papers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Label all papers in batch.
        
        Returns:
            (auto_labeled_papers, manual_review_papers)
        """
        logger.info(f"Starting batch labeling of {len(papers)} papers...")
        
        auto_labeled = []
        manual_review = []
        
        for i, paper in enumerate(papers):
            try:
                abstract = paper.get("Abstract", "")
                title = paper.get("Title", "")
                
                label, confidence, details = self.label_abstract(abstract, title)
                
                # Store results
                paper["confidence"] = round(confidence, 3)
                paper["scoring_details"] = details
                paper["labeled_at"] = datetime.now().isoformat()
                
                if label is not None:
                    # High confidence - auto-labeled
                    paper["Label"] = label
                    paper["label_source"] = "auto_high_confidence"
                    auto_labeled.append(paper)
                    self.auto_label_count += 1
                    
                    status = "✓ AUTO" if label == 1 else "- AUTO"
                    logger.debug(f"[{i+1}/{len(papers)}] {status} ({confidence:.2f}): {title[:60]}...")
                else:
                    # Low confidence - needs manual review
                    paper["Label"] = None
                    paper["label_source"] = "needs_manual_review"
                    manual_review.append(paper)
                    self.manual_review_count += 1
                    
                    logger.debug(f"[{i+1}/{len(papers)}] MANUAL ({confidence:.2f}): {title[:60]}...")
                
                self.labeled_count += 1
                
            except Exception as e:
                logger.error(f"Error labeling paper {i}: {e}")
                paper["Label"] = 0  # Default to negative if error
                paper["label_source"] = "error_default"
                auto_labeled.append(paper)
        
        logger.info(f"Batch labeling complete:")
        logger.info(f"  Auto-labeled: {len(auto_labeled)} ({len(auto_labeled)/len(papers)*100:.1f}%)")
        logger.info(f"  Manual review needed: {len(manual_review)} ({len(manual_review)/len(papers)*100:.1f}%)")
        
        return auto_labeled, manual_review

# ============================================================================
# MANUAL REVIEW INTERFACE
# ============================================================================

class ManualReviewInterface:
    """Interactive CLI for manual labeling."""
    
    def __init__(self, log_file: str = MANUAL_REVIEW_LOG):
        self.log_file = log_file
        self.reviewed_count = 0
        self.labeled_count = 0
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize review log file."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write(f"Manual Review Log - {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
    
    def _log_review(self, paper: Dict, label: int, notes: str = ""):
        """Log manual review decision."""
        with open(self.log_file, 'a') as f:
            f.write(f"DOI: {paper.get('DOI')}\n")
            f.write(f"Title: {paper.get('Title')}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Confidence (auto): {paper.get('confidence', 'N/A')}\n")
            if notes:
                f.write(f"Notes: {notes}\n")
            f.write("-"*80 + "\n\n")
    
    def _display_paper(self, paper: Dict, index: int, total: int):
        """Display paper information to user."""
        print("\n" + "="*80)
        print(f"[{index}/{total}] MANUAL REVIEW")
        print("="*80)
        print(f"Title: {paper.get('Title')}")
        print(f"Source: {paper.get('Source')} | Year: {paper.get('Year', 'N/A')}")
        print(f"Auto-score: {paper.get('confidence', 'N/A')}")
        print(f"DOI: {paper.get('DOI')}")
        print("-"*80)
        print("Abstract:")
        print(paper.get('Abstract', 'N/A')[:800])
        if len(paper.get('Abstract', '')) > 800:
            print("[... truncated]")
        print("-"*80)
    
    def _get_user_input(self) -> Tuple[Optional[int], str]:
        """Get user input for labeling."""
        while True:
            response = input("\nLabel [1=relevant/0=not relevant/s=skip/q=quit]? ").strip().lower()
            
            if response == "1":
                return 1, ""
            elif response == "0":
                return 0, ""
            elif response == "s":
                return None, "user_skip"
            elif response == "q":
                return -1, "user_quit"
            else:
                print("Invalid input. Enter: 1, 0, s, or q")
    
    def review_papers(self, papers: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Interactively review papers for manual labeling.
        
        Returns:
            (labeled_papers, quit_at_index)
        """
        logger.info(f"Starting manual review of {len(papers)} papers...")
        print(f"\n{'='*80}")
        print(f"MANUAL REVIEW INTERFACE - {len(papers)} papers")
        print(f"{'='*80}")
        
        labeled_papers = []
        quit_index = -1
        
        for i, paper in enumerate(papers):
            self._display_paper(paper, i + 1, len(papers))
            
            label, notes = self._get_user_input()
            
            if label == -1:  # User quit
                quit_index = i
                logger.info(f"User quit at paper {i+1}/{len(papers)}")
                break
            elif label is None:  # User skip
                logger.debug(f"Paper {i+1} skipped by user")
                continue
            else:
                paper["Label"] = label
                paper["label_source"] = "manual_review"
                paper["reviewed_at"] = datetime.now().isoformat()
                labeled_papers.append(paper)
                self.labeled_count += 1
                self._log_review(paper, label, notes)
                logger.info(f"Paper {i+1}: Labeled as {label}")
            
            self.reviewed_count += 1
        
        logger.info(f"Manual review complete: {self.labeled_count} papers labeled")
        return labeled_papers, quit_index

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

class LabelingWorkflow:
    """Orchestrates the complete labeling workflow."""
    
    def __init__(self):
        self.labeler = InflationAbstractLabeler()
        self.reviewer = ManualReviewInterface()
        self.all_labeled = []
    
    def _validate_input(self, papers: List[Dict]) -> None:
        """Validate input data."""
        if not papers:
            raise InputValidationError("No papers provided")
        
        for paper in papers:
            if not isinstance(paper, dict):
                raise InputValidationError("Invalid paper format")
            if "Abstract" not in paper:
                raise InputValidationError("Paper missing Abstract")
    
    def _load_papers(self, file_path: str) -> List[Dict]:
        """Load papers from JSON file."""
        if not os.path.exists(file_path):
            raise InputValidationError(f"File not found: {file_path}")
        
        logger.info(f"Loading papers from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        self._validate_input(papers)
        logger.info(f"Loaded {len(papers)} papers")
        return papers
    
    def _save_labeled_papers(self, papers: List[Dict], output_file: str) -> None:
        """Save labeled papers to JSON file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        logger.info(f"Saving {len(papers)} labeled papers to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        if not os.path.exists(output_file):
            raise IOError(f"Failed to write file: {output_file}")
        
        logger.info(f"Successfully saved to {output_file}")
    
    def _generate_statistics(self, papers: List[Dict]) -> Dict:
        """Generate labeling statistics."""
        stats = {
            "total_papers": len(papers),
            "label_1_count": sum(1 for p in papers if p.get("Label") == 1),
            "label_0_count": sum(1 for p in papers if p.get("Label") == 0),
            "label_none_count": sum(1 for p in papers if p.get("Label") is None),
            "auto_labeled_count": sum(1 for p in papers if p.get("label_source") == "auto_high_confidence"),
            "manual_labeled_count": sum(1 for p in papers if p.get("label_source") == "manual_review"),
            "avg_confidence": sum(p.get("confidence", 0) for p in papers) / len(papers) if papers else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if stats["total_papers"] > 0:
            stats["positive_percentage"] = round(stats["label_1_count"] / stats["total_papers"] * 100, 1)
            stats["negative_percentage"] = round(stats["label_0_count"] / stats["total_papers"] * 100, 1)
        
        return stats
    
    def _save_statistics(self, stats: Dict) -> None:
        """Save statistics to JSON file."""
        logger.info("Saving statistics...")
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {STATS_FILE}")
    
    def run(self, do_manual_review: bool = True) -> str:
        """Execute complete labeling workflow."""
        logger.info("="*80)
        logger.info("LABELING WORKFLOW START")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("="*80)
        
        try:
            # Phase 1: Load papers
            papers = self._load_papers(INPUT_FILE)
            
            # Phase 2: Auto-label
            logger.info("="*80)
            logger.info("PHASE 1: AUTO-LABELING")
            logger.info("="*80)
            auto_labeled, manual_review = self.labeler.label_batch(papers)
            
            # Phase 3: Manual review (optional)
            if do_manual_review and manual_review:
                logger.info("="*80)
                logger.info("PHASE 2: MANUAL REVIEW")
                logger.info("="*80)
                manually_labeled, quit_index = self.reviewer.review_papers(manual_review)
                self.all_labeled = auto_labeled + manually_labeled
            else:
                logger.info("Skipping manual review")
                # For low-confidence papers with no manual review, default to negative
                for paper in manual_review:
                    paper["Label"] = 0
                    paper["label_source"] = "auto_fallback_negative"
                self.all_labeled = auto_labeled + manual_review
            
            # Phase 4: Statistics
            logger.info("="*80)
            logger.info("PHASE 3: STATISTICS & SUMMARY")
            logger.info("="*80)
            stats = self._generate_statistics(self.all_labeled)
            
            logger.info(f"Total papers labeled: {stats['total_papers']}")
            logger.info(f"  Positive (Label=1): {stats['label_1_count']} ({stats.get('positive_percentage', 0)}%)")
            logger.info(f"  Negative (Label=0): {stats['label_0_count']} ({stats.get('negative_percentage', 0)}%)")
            logger.info(f"  Auto-labeled: {stats['auto_labeled_count']}")
            logger.info(f"  Manual-labeled: {stats['manual_labeled_count']}")
            logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
            
            self._save_statistics(stats)
            
            # Phase 5: Save results
            logger.info("="*80)
            logger.info("PHASE 4: SAVING RESULTS")
            logger.info("="*80)
            self._save_labeled_papers(self.all_labeled, OUTPUT_FILE)
            
            logger.info("="*80)
            logger.info("✓ LABELING WORKFLOW COMPLETE")
            logger.info("="*80)
            logger.info(f"Output file: {OUTPUT_FILE}")
            logger.info(f"Statistics file: {STATS_FILE}")
            logger.info(f"Review log: {MANUAL_REVIEW_LOG}")
            
            return OUTPUT_FILE
            
        except Exception as e:
            logger.critical(f"Workflow failed: {e}", exc_info=True)
            raise

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # Note: Set do_manual_review=False to skip interactive review
        workflow = LabelingWorkflow()
        output_file = workflow.run(do_manual_review=True)
        print(f"\n✓ SUCCESS: Labeling complete")
        print(f"Output: {output_file}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
