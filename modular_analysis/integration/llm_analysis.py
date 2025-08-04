#!/usr/bin/env python3
"""
llm_analysis.py

LLM-based analysis for cluster labeling and insights.
"""

from typing import List, Dict, Any, Optional

class LLMAnalyzer:
    """LLM-based analysis functionality."""
    
    def __init__(self, client=None):
        self.client = client
    
    def analyze_cluster_with_llm(self, representative_samples: List[str]) -> str:
        """
        Call LLM to label cluster with archetypal theme.
        
        Args:
            representative_samples: Representative life events for the cluster
            
        Returns:
            Archetypal theme description
        """
        if not self.client:
            return "No OpenAI API key found. (LLM analysis skipped)"
        
        prompt_template = (
            "You are an expert in criminal psychology and behavioral analysis.\n"
            "Given these representative life events of a serial killer, identify\n"
            "the archetypal pattern or theme they represent. Be concise and specific.\n\n"
            "Life events:\n{events}\n\nArchetypal theme:"
        )
        
        joined_events = "\n".join(representative_samples)
        prompt_text = prompt_template.format(events=joined_events)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in criminal psychology and behavioral analysis."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR in LLM] {e}")
            return "Unknown (LLM error)"
    
    def analyze_all_clusters(self, cluster_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze all clusters with LLM.
        
        Args:
            cluster_info: List of cluster information dictionaries
            
        Returns:
            Updated cluster information with archetypal themes
        """
        if not self.client:
            print("[INFO] Skipping LLM archetype labeling (no API key).")
            for cinfo in cluster_info:
                cinfo["archetypal_theme"] = "N/A (LLM disabled)"
            return cluster_info
        
        print("[INFO] Analyzing clusters with LLM...")
        
        for cinfo in cluster_info:
            theme = self.analyze_cluster_with_llm(cinfo["representative_samples"])
            cinfo["archetypal_theme"] = theme
            print(f"  Cluster {cinfo['cluster_id']}: {theme}")
        
        return cluster_info
    
    def generate_analysis_summary(self, insights: Dict[str, Dict[str, Any]], 
                                cluster_info: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive analysis summary using LLM.
        
        Args:
            insights: Conditional insights
            cluster_info: Cluster information
            
        Returns:
            Analysis summary
        """
        if not self.client:
            return "LLM summary not available (no API key)"
        
        # Prepare summary data
        significant_insights = [k for k, v in insights.items() if v.get('significant', False)]
        cluster_themes = [f"Cluster {c['cluster_id']}: {c.get('archetypal_theme', 'Unknown')}" 
                         for c in cluster_info]
        
        prompt = f"""
        You are an expert criminologist analyzing patterns in criminal behavior. 
        Based on the following analysis results, provide a comprehensive summary of key findings:

        CLUSTER THEMES:
        {chr(10).join(cluster_themes)}

        SIGNIFICANT CONDITIONAL PATTERNS ({len(significant_insights)} found):
        {chr(10).join(significant_insights[:10])}  # Limit to first 10

        Please provide:
        1. Key behavioral archetypes identified
        2. Most significant conditional patterns
        3. Implications for criminal profiling
        4. Recommendations for further analysis

        Keep the summary concise but insightful (max 500 words).
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR generating summary] {e}")
            return f"Error generating summary: {e}"
