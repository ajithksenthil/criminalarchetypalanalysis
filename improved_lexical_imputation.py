#!/usr/bin/env python3
"""
improved_lexical_imputation.py

Improved lexical imputation with name standardization and better prompting.
"""

import re
import numpy as np
import openai
import os
from typing import List, Tuple


class ImprovedLexicalImputation:
    """
    Improved lexical imputation that standardizes names and provides better guidance
    for generating alternative event representations.
    """
    
    def __init__(self, client=None):
        """Initialize with OpenAI client."""
        self.client = client
        if self.client is None and os.environ.get('OPENAI_API_KEY'):
            self.client = openai.OpenAI()
        
        # Common name patterns to standardize
        self.name_patterns = [
            # First names
            r'\b(?:John|James|Robert|Michael|William|David|Richard|Joseph|Thomas|Charles|Christopher|Daniel|Matthew|Anthony|Donald|Steven|Paul|Andrew|Joshua|Kenneth|Kevin|Brian|George|Edward|Ronald|Timothy|Jason|Jeffrey|Ryan|Jacob|Gary|Nicholas|Eric|Jonathan|Stephen|Larry|Justin|Scott|Brandon|Benjamin|Samuel|Frank|Gregory|Raymond|Alexander|Patrick|Jack|Dennis|Jerry|Tyler|Aaron|Jose|Nathan|Henry|Zachary|Douglas|Peter|Adam|Harold|Carl|Arthur|Gerald|Roger|Keith|Jeremy|Lawrence|Terry|Austin|Sean|Christian|Albert|Joe|Juan|Elijah|Willie|Wayne|Ralph|Roy|Eugene|Russell|Louis|Philip|Johnny|Randy|Howard|Eugene|Russell|Louis|Philip|Johnny|Randy|Howard)\b',
            # Common female names
            r'\b(?:Mary|Patricia|Jennifer|Linda|Elizabeth|Barbara|Susan|Jessica|Sarah|Karen|Nancy|Lisa|Betty|Dorothy|Sandra|Ashley|Kimberly|Donna|Emily|Michelle|Carol|Amanda|Melissa|Deborah|Stephanie|Rebecca|Laura|Sharon|Cynthia|Kathleen|Amy|Shirley|Angela|Helen|Anna|Brenda|Pamela|Nicole|Samantha|Katherine|Emma|Ruth|Christine|Catherine|Debra|Rachel|Carolyn|Janet|Maria|Heather|Diane|Olivia|Julie|Joyce|Virginia|Victoria|Kelly|Lauren|Christina|Joan|Evelyn|Judith|Megan|Cheryl|Andrea|Hannah|Martha|Madison|Teresa|Gloria|Sara|Janice|Jean|Alice|Kathryn|Frances|Judy|Isabella|Beverly|Denise|Danielle|Marilyn|Amber|Theresa|Sophia|Marie|Diana|Brittany|Natalie|Charlotte|Rose|Alexis|Kayla)\b',
            # Last names
            r'\b(?:Smith|Johnson|Williams|Brown|Jones|Garcia|Miller|Davis|Rodriguez|Martinez|Hernandez|Lopez|Gonzalez|Wilson|Anderson|Thomas|Taylor|Moore|Jackson|Martin|Lee|Perez|Thompson|White|Harris|Sanchez|Clark|Ramirez|Lewis|Robinson|Walker|Young|Allen|King|Wright|Scott|Torres|Nguyen|Hill|Flores|Green|Adams|Nelson|Baker|Hall|Rivera|Campbell|Mitchell|Carter|Roberts|Gomez|Phillips|Evans|Turner|Diaz|Parker|Cruz|Edwards|Collins|Reyes|Stewart|Morris|Morales|Murphy|Cook|Rogers|Gutierrez|Ortiz|Morgan|Cooper|Peterson|Bailey|Reed|Kelly|Howard|Ramos|Kim|Cox|Ward|Richardson|Watson|Brooks|Chavez|Wood|James|Bennett|Gray|Mendoza|Ruiz|Hughes|Price|Alvarez|Castillo|Sanders|Patel|Myers|Long|Ross|Foster|Jimenez)\b',
            # Nicknames and variations
            r'\b(?:Bob|Bobby|Rob|Robbie|Bill|Billy|Will|Dick|Rick|Ricky|Jim|Jimmy|Mike|Mikey|Tom|Tommy|Chuck|Charlie|Chris|Dan|Danny|Matt|Tony|Don|Donny|Steve|Andy|Josh|Ken|Kenny|Dave|Davy)\b',
        ]
        
        # Location patterns
        self.location_patterns = [
            r'\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b',
            r'\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b',
        ]
    
    def standardize_names(self, text: str) -> Tuple[str, List[str]]:
        """
        Replace all names with standardized placeholders.
        
        Returns:
            Tuple of (standardized_text, list_of_replaced_names)
        """
        standardized = text
        replaced_names = []
        
        # Replace names with PERSON placeholder
        for pattern in self.name_patterns:
            matches = re.finditer(pattern, standardized, re.IGNORECASE)
            for match in reversed(list(matches)):
                name = match.group()
                replaced_names.append(name)
                standardized = standardized[:match.start()] + "[PERSON]" + standardized[match.end():]
        
        # Replace locations with LOCATION placeholder
        for pattern in self.location_patterns:
            matches = re.finditer(pattern, standardized, re.IGNORECASE)
            for match in reversed(list(matches)):
                location = match.group()
                replaced_names.append(location)
                standardized = standardized[:match.start()] + "[LOCATION]" + standardized[match.end():]
        
        # Clean up multiple placeholders
        standardized = re.sub(r'\[PERSON\]\s+\[PERSON\]', '[PERSON]', standardized)
        standardized = re.sub(r'\s+', ' ', standardized).strip()
        
        return standardized, replaced_names
    
    def generate_improved_variations(self, text: str, num_variants: int = 5) -> List[str]:
        """
        Generate lexical variations with improved prompting and name standardization.
        """
        # First, standardize names
        standardized_text, replaced_names = self.standardize_names(text)
        
        if not self.client:
            # If no client, just return the standardized version
            return [standardized_text]
        
        # Improved prompt with better guidance
        prompt = f"""You are helping to standardize criminal life event descriptions for analysis.

Original event: {text}
Standardized event: {standardized_text}

Generate {num_variants} alternative phrasings of this event that:
1. Preserve the exact same meaning and criminal behavior
2. Keep all [PERSON] and [LOCATION] placeholders unchanged
3. Vary the sentence structure and word choice
4. Use different synonyms for actions and descriptors
5. Maintain the same level of formality
6. Do not add or remove any factual information

Important: Focus on the ACTION and BEHAVIOR, not the specific names or places.

Alternative versions:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at rephrasing criminal behavior descriptions while maintaining exact meaning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content.strip()
            variations = [line.strip() for line in reply.split("\n") if line.strip() and not line.strip().startswith(("#", "-", "*"))]
            
            # Ensure we have the right number of variations
            if len(variations) < num_variants:
                variations.extend([standardized_text] * (num_variants - len(variations)))
            elif len(variations) > num_variants:
                variations = variations[:num_variants]
            
            # Always include the standardized original
            variations.append(standardized_text)
            
            return variations
            
        except Exception as e:
            print(f"[ERROR] Generating variations: {e}")
            return [standardized_text]
    
    def get_improved_embedding(self, event_text: str, model, num_variants: int = 5) -> np.ndarray:
        """
        Get improved embedding with name standardization and better variations.
        """
        # Generate variations with standardized names
        variations = self.generate_improved_variations(event_text, num_variants)
        
        # Compute embeddings for all variations
        embeddings = model.encode(variations)
        
        # Return the average (centroid) embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        return avg_embedding


def update_analysis_to_use_improved_imputation():
    """
    Create a modified version of the analysis that uses improved imputation.
    """
    code = '''
# Replace the original generate_lexical_variations function with:

from improved_lexical_imputation import ImprovedLexicalImputation

# Initialize the improved imputation
imputer = ImprovedLexicalImputation(client=client)

def generate_lexical_variations(text, num_variants=5):
    """Use improved lexical variations with name standardization."""
    return imputer.generate_improved_variations(text, num_variants)

# Or replace get_imputed_embedding with:
def get_imputed_embedding(event_text, model, num_variants=5):
    """Get embedding using improved imputation."""
    return imputer.get_improved_embedding(event_text, model, num_variants)
'''
    
    print("To use improved imputation, add this to analysis_integration_improved.py:")
    print(code)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Improved Lexical Imputation")
    print("="*60)
    
    # Test cases
    test_events = [
        "John Smith was arrested for assault in California",
        "Robert murdered his wife Mary in Texas", 
        "Dale moved to Colorado where his mother was hospitalized",
        "William Johnson dealt drugs in New York before being caught by police",
        "The defendant James Brown was convicted of robbery in Miami"
    ]
    
    imputer = ImprovedLexicalImputation()
    
    for event in test_events:
        print(f"\nOriginal: {event}")
        standardized, names = imputer.standardize_names(event)
        print(f"Standardized: {standardized}")
        print(f"Replaced: {names}")
        
        # Generate variations
        if imputer.client:
            variations = imputer.generate_improved_variations(event, num_variants=3)
            print("Variations:")
            for i, var in enumerate(variations, 1):
                print(f"  {i}. {var}")
        else:
            print("(No OpenAI client - would generate variations here)")
    
    print("\n" + "="*60)
    print("Benefits of this approach:")
    print("1. Names don't bias the embeddings")
    print("2. Focus is on criminal behavior patterns, not individuals")
    print("3. Better clustering of similar events")
    print("4. More meaningful archetypes")
    
    update_analysis_to_use_improved_imputation()