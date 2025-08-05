#!/usr/bin/env python3
"""
lexical_bias_processor.py

Comprehensive lexical bias processing for criminal life events.
Removes specific names, locations, and other identifying information
to focus on behavioral patterns rather than specific individuals.
"""

import re
import spacy
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path

class LexicalBiasProcessor:
    """Process text to remove lexical bias from criminal life events."""
    
    def __init__(self):
        """Initialize the processor with NLP models and entity mappings."""
        try:
            # Load spaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[WARNING] spaCy model 'en_core_web_sm' not found.")
            print("Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define entity replacement mappings
        self.entity_replacements = {
            'PERSON': '[PERSON]',
            'GPE': '[LOCATION]',  # Geopolitical entity (cities, countries)
            'LOC': '[LOCATION]',  # Locations
            'ORG': '[ORGANIZATION]',  # Organizations
            'DATE': '[DATE]',
            'TIME': '[TIME]',
            'MONEY': '[AMOUNT]',
            'CARDINAL': '[NUMBER]',  # Cardinal numbers
            'ORDINAL': '[NUMBER]',   # Ordinal numbers
            'FAC': '[FACILITY]',     # Facilities (buildings, airports, etc.)
            'EVENT': '[EVENT]',      # Named events
            'LAW': '[LAW]',          # Legal documents
            'LANGUAGE': '[LANGUAGE]',
            'NORP': '[GROUP]',       # Nationalities, religious groups
            'PRODUCT': '[PRODUCT]',  # Products
            'WORK_OF_ART': '[MEDIA]' # Books, songs, etc.
        }
        
        # Common criminal justice terms to preserve
        self.preserve_terms = {
            'murder', 'kill', 'death', 'victim', 'crime', 'arrest', 'prison', 'jail',
            'trial', 'court', 'judge', 'jury', 'guilty', 'innocent', 'sentence',
            'evidence', 'witness', 'testimony', 'conviction', 'acquittal',
            'police', 'detective', 'investigation', 'forensic', 'DNA',
            'rape', 'assault', 'robbery', 'burglary', 'theft', 'fraud',
            'weapon', 'gun', 'knife', 'violence', 'abuse', 'torture',
            'serial killer', 'psychopath', 'sociopath', 'mental health',
            'childhood', 'family', 'parent', 'mother', 'father', 'sibling',
            'school', 'education', 'work', 'job', 'employment', 'relationship'
        }
        
        # Patterns for additional cleaning
        self.cleaning_patterns = [
            # Remove specific case references
            (r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+\b', '[CASE]'),
            # Remove specific law enforcement agencies
            (r'\b(FBI|CIA|DEA|ATF|SWAT)\b', '[AGENCY]'),
            # Remove specific court names
            (r'\b[A-Z][a-z]+ (County|District|Superior|Supreme) Court\b', '[COURT]'),
            # Remove specific prison names
            (r'\b[A-Z][a-z]+ (State )?Prison\b', '[PRISON]'),
            (r'\b[A-Z][a-z]+ Correctional (Facility|Institution)\b', '[PRISON]'),
            # Remove specific hospital/medical facility names
            (r'\b[A-Z][a-z]+ (Hospital|Medical Center|Clinic)\b', '[MEDICAL_FACILITY]'),
            # Remove specific school names
            (r'\b[A-Z][a-z]+ (Elementary|Middle|High) School\b', '[SCHOOL]'),
            (r'\b[A-Z][a-z]+ University\b', '[UNIVERSITY]'),
            # Remove specific addresses
            (r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)\b', '[ADDRESS]'),
            # Remove phone numbers
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            # Remove social security numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            # Remove case numbers
            (r'\bCase No\. \d+\b', '[CASE_NUMBER]'),
            (r'\bDocket No\. \d+\b', '[CASE_NUMBER]'),
        ]
    
    def process_text(self, text: str) -> str:
        """
        Process a single text to remove lexical bias.
        
        Args:
            text: Input text
            
        Returns:
            Processed text with entities replaced
        """
        if not text or not text.strip():
            return text
        
        processed_text = text
        
        # Apply spaCy NER if available
        if self.nlp:
            processed_text = self._apply_ner_replacement(processed_text)
        
        # Apply pattern-based cleaning
        processed_text = self._apply_pattern_cleaning(processed_text)
        
        # Clean up multiple spaces and formatting
        processed_text = self._clean_formatting(processed_text)
        
        return processed_text
    
    def _apply_ner_replacement(self, text: str) -> str:
        """Apply Named Entity Recognition replacement."""
        doc = self.nlp(text)
        
        # Sort entities by start position (reverse order for replacement)
        entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)
        
        processed_text = text
        
        for ent in entities:
            entity_text = ent.text.lower()
            
            # Skip if it's a preserved term
            if any(term in entity_text for term in self.preserve_terms):
                continue
            
            # Skip very short entities (likely not meaningful)
            if len(ent.text.strip()) < 2:
                continue
            
            # Replace with generic label
            if ent.label_ in self.entity_replacements:
                replacement = self.entity_replacements[ent.label_]
                processed_text = (processed_text[:ent.start_char] + 
                                replacement + 
                                processed_text[ent.end_char:])
        
        return processed_text
    
    def _apply_pattern_cleaning(self, text: str) -> str:
        """Apply pattern-based cleaning."""
        processed_text = text
        
        for pattern, replacement in self.cleaning_patterns:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting issues."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple consecutive entity labels
        text = re.sub(r'(\[[A-Z_]+\])\s*\1+', r'\1', text)
        
        # Clean up punctuation around entity labels
        text = re.sub(r'\s*\[([A-Z_]+)\]\s*', r' [\1] ', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s+', r'\1 ', text)
        
        return text.strip()
    
    def process_event_list(self, events: List[str]) -> List[str]:
        """
        Process a list of criminal life events.
        
        Args:
            events: List of life event descriptions
            
        Returns:
            List of processed events
        """
        print(f"[INFO] Processing {len(events)} events for lexical bias removal...")
        
        processed_events = []
        
        for i, event in enumerate(events):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(events)} events")
            
            processed_event = self.process_text(event)
            processed_events.append(processed_event)
        
        print(f"[INFO] Lexical bias processing complete")
        return processed_events
    
    def analyze_bias_reduction(self, original_events: List[str], 
                             processed_events: List[str]) -> Dict[str, any]:
        """
        Analyze the effectiveness of bias reduction.
        
        Args:
            original_events: Original event descriptions
            processed_events: Processed event descriptions
            
        Returns:
            Analysis results
        """
        if not self.nlp:
            return {"error": "spaCy not available for analysis"}
        
        original_entities = set()
        processed_entities = set()
        
        # Extract entities from original texts
        for event in original_events[:100]:  # Sample for efficiency
            doc = self.nlp(event)
            for ent in doc.ents:
                if ent.label_ in self.entity_replacements:
                    original_entities.add((ent.text, ent.label_))
        
        # Extract entities from processed texts
        for event in processed_events[:100]:
            doc = self.nlp(event)
            for ent in doc.ents:
                if ent.label_ in self.entity_replacements:
                    processed_entities.add((ent.text, ent.label_))
        
        # Count entity types
        original_by_type = {}
        for text, label in original_entities:
            original_by_type[label] = original_by_type.get(label, 0) + 1
        
        processed_by_type = {}
        for text, label in processed_entities:
            processed_by_type[label] = processed_by_type.get(label, 0) + 1
        
        # Calculate reduction
        total_original = len(original_entities)
        total_processed = len(processed_entities)
        reduction_rate = (total_original - total_processed) / total_original if total_original > 0 else 0
        
        return {
            "original_entity_count": total_original,
            "processed_entity_count": total_processed,
            "reduction_rate": reduction_rate,
            "original_by_type": original_by_type,
            "processed_by_type": processed_by_type,
            "sample_replacements": self._get_sample_replacements(original_events[:10], processed_events[:10])
        }
    
    def _get_sample_replacements(self, original: List[str], processed: List[str]) -> List[Dict[str, str]]:
        """Get sample before/after replacements for inspection."""
        samples = []
        
        for orig, proc in zip(original[:5], processed[:5]):
            if orig != proc:
                samples.append({
                    "original": orig[:200] + "..." if len(orig) > 200 else orig,
                    "processed": proc[:200] + "..." if len(proc) > 200 else proc
                })
        
        return samples

def test_lexical_bias_processor():
    """Test the lexical bias processor with sample criminal events."""
    
    processor = LexicalBiasProcessor()
    
    # Sample criminal life events with bias
    sample_events = [
        "Charles Albright was born in Amarillo, Texas on August 10, 1933",
        "Defense Forensic expert, Samuel J. Palenik, testified that hair samples may not belong to Albright",
        "Jury convicts Albright on one murder, Shirley Williams, due to strongest evidence",
        "Born in West Orange, New Jersey: Last of 8 children including 5 sisters and 2 brothers",
        "Served 3 years in Folsom State Prison for armed robbery",
        "Victim Ten-Cindy Hudspeth, a 20 year-old college student from UCLA",
        "Trial held at Los Angeles County Superior Court with Judge Robert Martinez presiding"
    ]
    
    print("TESTING LEXICAL BIAS PROCESSOR")
    print("=" * 50)
    
    print("\nORIGINAL vs PROCESSED EVENTS:")
    print("-" * 30)
    
    for i, event in enumerate(sample_events):
        processed = processor.process_text(event)
        print(f"\n{i+1}. ORIGINAL:")
        print(f"   {event}")
        print(f"   PROCESSED:")
        print(f"   {processed}")
    
    # Analyze bias reduction
    processed_events = [processor.process_text(event) for event in sample_events]
    analysis = processor.analyze_bias_reduction(sample_events, processed_events)
    
    print(f"\nBIAS REDUCTION ANALYSIS:")
    print(f"Original entities: {analysis.get('original_entity_count', 'N/A')}")
    print(f"Processed entities: {analysis.get('processed_entity_count', 'N/A')}")
    print(f"Reduction rate: {analysis.get('reduction_rate', 0):.1%}")

if __name__ == "__main__":
    test_lexical_bias_processor()
