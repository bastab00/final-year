"""
Production-grade script to validate, deduplicate, and merge datasets.

This script:
- Validates labeled dataset integrity
- Deduplicates against original dataset
- Merges into expanded dataset
- Verifies final dataset meets specifications
- Backs up original data
"""

import json
import os
import sys
import logging
import shutil
from typing import Dict, List, Tuple, Set
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

ORIGINAL_FILE = "data/raw/inflation_dataset.json"
LABELED_FILE = "data/raw/collected_abstracts_labeled.json"
OUTPUT_FILE = "data/raw/inflation_dataset_expanded.json"
BACKUP_DIR = "data/raw/backups"
MERGE_REPORT_FILE = "data/raw/merge_report.json"

# Expected dataset sizes
ORIGINAL_SIZE = 1134
NEW_SIZE_TARGET = 4000  # Target ~4000 new papers
FINAL_SIZE_TARGET = ORIGINAL_SIZE + NEW_SIZE_TARGET  # ~5134

# Acceptable variance (±X papers)
SIZE_VARIANCE = 200

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class MergeError(Exception):
    """Base exception for merge errors."""
    pass

class ValidationError(MergeError):
    """Data validation error."""
    pass

class IOError(MergeError):
    """I/O error."""
    pass

# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    """Validates dataset integrity."""
    
    def validate_paper(self, paper: Dict, index: int = -1) -> Tuple[bool, str]:
        """
        Validate individual paper.
        
        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        required_fields = ["DOI", "Abstract", "Label"]
        for field in required_fields:
            if field not in paper:
                return False, f"Paper {index}: Missing required field '{field}'"
        
        # Validate DOI
        doi = paper.get("DOI")
        if not isinstance(doi, str) or not doi.strip():
            return False, f"Paper {index}: Invalid DOI (must be non-empty string)"
        
        # Validate Abstract
        abstract = paper.get("Abstract")
        if not isinstance(abstract, str) or len(abstract.strip()) < 50:
            return False, f"Paper {index}: Abstract too short (min 50 chars)"
        
        # Validate Label
        label = paper.get("Label")
        if label is not None and label not in [0, 1]:
            return False, f"Paper {index}: Invalid Label (must be 0, 1, or None)"
        
        # Validate Title (if present)
        if "Title" in paper:
            title = paper.get("Title")
            if not isinstance(title, str) or not title.strip():
                return False, f"Paper {index}: Invalid Title"
        
        # Validate Year (if present)
        if "Year" in paper:
            year = paper.get("Year")
            if year is not None and (not isinstance(year, int) or year < 1900 or year > 2030):
                return False, f"Paper {index}: Invalid Year"
        
        return True, ""
    
    def validate_dataset(self, papers: List[Dict], dataset_name: str = "dataset") -> Dict:
        """
        Validate entire dataset.
        
        Returns:
            validation_report_dict
        """
        logger.info(f"Validating {dataset_name} ({len(papers)} papers)...")
        
        report = {
            "dataset_name": dataset_name,
            "total_papers": len(papers),
            "valid_papers": 0,
            "invalid_papers": 0,
            "errors": [],
            "label_distribution": {"0": 0, "1": 0, "None": 0},
            "with_title": 0,
            "with_year": 0,
            "with_authors": 0,
            "unique_sources": set(),
            "timestamp": datetime.now().isoformat()
        }
        
        for i, paper in enumerate(papers):
            is_valid, error_msg = self.validate_paper(paper, i)
            
            if not is_valid:
                report["invalid_papers"] += 1
                report["errors"].append(error_msg)
                logger.warning(error_msg)
            else:
                report["valid_papers"] += 1
            
            # Count label distribution
            label = str(paper.get("Label"))
            report["label_distribution"][label] = report["label_distribution"].get(label, 0) + 1
            
            # Count optional fields
            if "Title" in paper and paper["Title"]:
                report["with_title"] += 1
            if "Year" in paper and paper["Year"]:
                report["with_year"] += 1
            if "Authors" in paper and paper["Authors"]:
                report["with_authors"] += 1
            if "Source" in paper:
                report["unique_sources"].add(paper["Source"])
        
        # Convert set to list for JSON serialization
        report["unique_sources"] = list(report["unique_sources"])
        
        # Summary
        logger.info(f"  Valid papers: {report['valid_papers']}/{report['total_papers']}")
        logger.info(f"  Invalid papers: {report['invalid_papers']}")
        logger.info(f"  Label distribution: {report['label_distribution']}")
        
        if report["invalid_papers"] > 0:
            logger.error(f"  Found {report['invalid_papers']} invalid papers!")
            raise ValidationError(f"Validation failed: {report['invalid_papers']} invalid papers")
        
        return report

# ============================================================================
# DEDUPLICATION ENGINE
# ============================================================================

class DeduplicationEngine:
    """Handles deduplication logic."""
    
    def __init__(self):
        self.seen_dois: Set[str] = set()
        self.seen_titles: Set[str] = set()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        return " ".join(title.lower().split())[:100]
    
    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI for comparison."""
        return doi.lower().strip()
    
    def build_index_from_dataset(self, papers: List[Dict]) -> None:
        """Build index of DOIs and titles from dataset."""
        logger.info(f"Building deduplication index from {len(papers)} papers...")
        
        for paper in papers:
            doi = paper.get("DOI", "")
            title = paper.get("Title", "")
            
            if doi:
                self.seen_dois.add(self._normalize_doi(doi))
            if title:
                self.seen_titles.add(self._normalize_title(title))
        
        logger.info(f"Index built: {len(self.seen_dois)} DOIs, {len(self.seen_titles)} titles")
    
    def is_duplicate(self, paper: Dict) -> Tuple[bool, str]:
        """
        Check if paper is duplicate.
        
        Returns:
            (is_duplicate, reason)
        """
        doi = paper.get("DOI", "")
        title = paper.get("Title", "")
        
        # Check DOI
        if doi and self._normalize_doi(doi) in self.seen_dois:
            return True, f"Duplicate DOI: {doi}"
        
        # Check title
        if title and self._normalize_title(title) in self.seen_titles:
            return True, f"Duplicate title: {title[:50]}..."
        
        return False, ""
    
    def filter_duplicates(self, papers: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Filter out duplicates from papers list.
        
        Returns:
            (unique_papers, duplicates_removed_count)
        """
        logger.info(f"Filtering duplicates from {len(papers)} papers...")
        
        unique_papers = []
        duplicates_removed = 0
        
        for paper in papers:
            is_dup, reason = self.is_duplicate(paper)
            if is_dup:
                duplicates_removed += 1
                logger.debug(f"  Removed: {reason}")
            else:
                # Add to index
                doi = paper.get("DOI", "")
                title = paper.get("Title", "")
                if doi:
                    self.seen_dois.add(self._normalize_doi(doi))
                if title:
                    self.seen_titles.add(self._normalize_title(title))
                
                unique_papers.append(paper)
        
        logger.info(f"Filtering complete: {duplicates_removed} duplicates removed, {len(unique_papers)} unique papers")
        return unique_papers, duplicates_removed

# ============================================================================
# MERGE ENGINE
# ============================================================================

class MergeEngine:
    """Orchestrates dataset merging."""
    
    def __init__(self):
        self.validator = DataValidator()
        self.deduplicator = DeduplicationEngine()
        self.original_papers: List[Dict] = []
        self.new_papers: List[Dict] = []
        self.merged_papers: List[Dict] = []
        self.merge_report: Dict = {}
    
    def _load_json_file(self, file_path: str, dataset_name: str) -> List[Dict]:
        """Load JSON file with error handling."""
        if not os.path.exists(file_path):
            raise IOError(f"File not found: {file_path}")
        
        logger.info(f"Loading {dataset_name} from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValidationError(f"{dataset_name} is not a list")
            
            logger.info(f"Loaded {len(data)} papers from {dataset_name}")
            return data
        
        except json.JSONDecodeError as e:
            raise IOError(f"JSON decode error in {file_path}: {e}")
        except Exception as e:
            raise IOError(f"Error loading {file_path}: {e}")
    
    def _save_json_file(self, data: List[Dict], file_path: str) -> None:
        """Save JSON file with error handling."""
        try:
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if not os.path.exists(file_path):
                raise IOError(f"File was not created: {file_path}")
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Successfully saved to {file_path} ({file_size} bytes)")
        
        except Exception as e:
            raise IOError(f"Error saving to {file_path}: {e}")
    
    def _backup_original(self) -> None:
        """Backup original file."""
        if not os.path.exists(ORIGINAL_FILE):
            logger.warning("Original file does not exist, skipping backup")
            return
        
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"inflation_dataset_backup_{timestamp}.json")
        
        logger.info(f"Creating backup of original file...")
        try:
            shutil.copy2(ORIGINAL_FILE, backup_file)
            logger.info(f"Backup created: {backup_file}")
            self.merge_report["backup_file"] = backup_file
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise IOError(f"Backup creation failed: {e}")
    
    def merge(self) -> str:
        """Execute complete merge workflow."""
        logger.info("="*80)
        logger.info("MERGE WORKFLOW START")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("="*80)
        
        try:
            # Phase 1: Load datasets
            logger.info("="*80)
            logger.info("PHASE 1: LOADING DATASETS")
            logger.info("="*80)
            
            self.original_papers = self._load_json_file(ORIGINAL_FILE, "original dataset")
            self.new_papers = self._load_json_file(LABELED_FILE, "new labeled dataset")
            
            self.merge_report["load_timestamp"] = datetime.now().isoformat()
            self.merge_report["original_papers_loaded"] = len(self.original_papers)
            self.merge_report["new_papers_loaded"] = len(self.new_papers)
            
            # Phase 2: Validate original dataset
            logger.info("="*80)
            logger.info("PHASE 2: VALIDATING ORIGINAL DATASET")
            logger.info("="*80)
            
            original_validation = self.validator.validate_dataset(self.original_papers, "original dataset")
            self.merge_report["original_validation"] = original_validation
            
            # Phase 3: Validate new dataset
            logger.info("="*80)
            logger.info("PHASE 3: VALIDATING NEW DATASET")
            logger.info("="*80)
            
            new_validation = self.validator.validate_dataset(self.new_papers, "new dataset")
            self.merge_report["new_validation"] = new_validation
            
            # Phase 4: Deduplicate
            logger.info("="*80)
            logger.info("PHASE 4: DEDUPLICATION")
            logger.info("="*80)
            
            # Build index from original
            self.deduplicator.build_index_from_dataset(self.original_papers)
            
            # Filter duplicates from new papers
            filtered_new, duplicates_found = self.deduplicator.filter_duplicates(self.new_papers)
            
            self.merge_report["duplicates_removed"] = duplicates_found
            self.merge_report["unique_new_papers"] = len(filtered_new)
            
            # Phase 5: Merge
            logger.info("="*80)
            logger.info("PHASE 5: MERGING DATASETS")
            logger.info("="*80)
            
            self.merged_papers = self.original_papers + filtered_new
            logger.info(f"Merged dataset size: {len(self.merged_papers)}")
            self.merge_report["merged_papers_count"] = len(self.merged_papers)
            
            # Phase 6: Validate merged dataset
            logger.info("="*80)
            logger.info("PHASE 6: VALIDATING MERGED DATASET")
            logger.info("="*80)
            
            merged_validation = self.validator.validate_dataset(self.merged_papers, "merged dataset")
            self.merge_report["merged_validation"] = merged_validation
            
            # Phase 7: Size verification
            logger.info("="*80)
            logger.info("PHASE 7: SIZE VERIFICATION")
            logger.info("="*80)
            
            merged_size = len(self.merged_papers)
            expected_min = FINAL_SIZE_TARGET - SIZE_VARIANCE
            expected_max = FINAL_SIZE_TARGET + SIZE_VARIANCE
            
            logger.info(f"Merged dataset size: {merged_size}")
            logger.info(f"Expected range: {expected_min} - {expected_max}")
            
            if merged_size < expected_min or merged_size > expected_max:
                logger.warning(f"Merged dataset size {merged_size} outside expected range {expected_min}-{expected_max}")
                # Don't fail, just warn
            else:
                logger.info("✓ Dataset size within acceptable range")
            
            self.merge_report["size_verification"] = {
                "actual": merged_size,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "passed": expected_min <= merged_size <= expected_max
            }
            
            # Phase 8: Label distribution check
            logger.info("="*80)
            logger.info("PHASE 8: LABEL DISTRIBUTION ANALYSIS")
            logger.info("="*80)
            
            label_1_count = sum(1 for p in self.merged_papers if p.get("Label") == 1)
            label_0_count = sum(1 for p in self.merged_papers if p.get("Label") == 0)
            total = len(self.merged_papers)
            
            positive_pct = label_1_count / total * 100 if total > 0 else 0
            negative_pct = label_0_count / total * 100 if total > 0 else 0
            
            logger.info(f"Label distribution:")
            logger.info(f"  Positive (1): {label_1_count} ({positive_pct:.1f}%)")
            logger.info(f"  Negative (0): {label_0_count} ({negative_pct:.1f}%)")
            
            # Target: ~60% positive
            if 50 < positive_pct < 70:
                logger.info("✓ Label distribution within expected range (50-70% positive)")
            else:
                logger.warning(f"Label distribution outside target range: {positive_pct:.1f}% positive")
            
            self.merge_report["label_distribution"] = {
                "positive_count": label_1_count,
                "negative_count": label_0_count,
                "positive_percentage": round(positive_pct, 1),
                "negative_percentage": round(negative_pct, 1)
            }
            
            # Phase 9: Backup original
            logger.info("="*80)
            logger.info("PHASE 9: BACKUP ORIGINAL")
            logger.info("="*80)
            
            self._backup_original()
            
            # Phase 10: Save merged dataset
            logger.info("="*80)
            logger.info("PHASE 10: SAVING MERGED DATASET")
            logger.info("="*80)
            
            self._save_json_file(self.merged_papers, OUTPUT_FILE)
            self.merge_report["output_file"] = OUTPUT_FILE
            
            # Phase 11: Save merge report
            logger.info("="*80)
            logger.info("PHASE 11: SAVING MERGE REPORT")
            logger.info("="*80)
            
            self.merge_report["completion_timestamp"] = datetime.now().isoformat()
            with open(MERGE_REPORT_FILE, 'w') as f:
                json.dump(self.merge_report, f, indent=2)
            logger.info(f"Merge report saved to {MERGE_REPORT_FILE}")
            
            # Final summary
            logger.info("="*80)
            logger.info("✓ MERGE WORKFLOW COMPLETE")
            logger.info("="*80)
            logger.info(f"Original: {len(self.original_papers)} papers")
            logger.info(f"New: {len(filtered_new)} papers (after dedup)")
            logger.info(f"Merged: {len(self.merged_papers)} papers")
            logger.info(f"Output: {OUTPUT_FILE}")
            logger.info(f"Report: {MERGE_REPORT_FILE}")
            
            return OUTPUT_FILE
        
        except Exception as e:
            logger.critical(f"Merge failed: {e}", exc_info=True)
            raise

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        engine = MergeEngine()
        output_file = engine.merge()
        print(f"\n✓ SUCCESS: Merge and validation complete")
        print(f"Output: {output_file}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
