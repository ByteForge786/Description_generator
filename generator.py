import pandas as pd
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import json
from test import chinou_response
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attribute_description.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AttributeDescriptionGenerator:
    def __init__(self, input_folder: str = "input_data", output_folder: str = "output_data", 
                 batch_size: int = 3, max_workers: int = 4):
        """
        Initialize the generator with configuration parameters.
        
        Args:
            input_folder: Folder containing input CSV files
            output_folder: Folder for output CSV files
            batch_size: Number of attributes to process in one LLM call
            max_workers: Maximum number of parallel threads
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.setup_folders()
        
    def setup_folders(self):
        """Create input and output folders if they don't exist."""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_sample_values(self, df: pd.DataFrame, column: str) -> str:
        """
        Get representative sample values for a column.
        
        Args:
            df: DataFrame containing the data
            column: Column name to get samples from
        
        Returns:
            String containing sample values
        """
        try:
            # Remove null values and duplicates
            samples = df[column].dropna().drop_duplicates().astype(str)
            
            if not len(samples):
                return "NO_SAMPLES_AVAILABLE"
            
            # Check if any sample has more than 40 words
            long_descriptions = any(len(str(sample).split()) > 40 for sample in samples)
            
            if long_descriptions:
                # Return just one sample for long descriptions
                return str(samples.iloc[0])
            else:
                # Return up to 5 samples
                return ", ".join(samples.head(5).tolist())
                
        except Exception as e:
            logger.error(f"Error getting samples for column {column}: {str(e)}")
            return "ERROR_GETTING_SAMPLES"

    def create_prompt(self, attributes: List[Tuple[str, str]], company_context: str = "") -> str:
        """
        Create prompt for the LLM with attribute names and their samples.
        
        Args:
            attributes: List of tuples containing (attribute_name, sample_values)
            company_context: Additional company-specific context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the company context and sample values, generate specific and detailed descriptions for the following attributes.
Company Context: {company_context}

For each attribute, provide the description in the following JSON format:
{{
    "attributes": [
        {{
            "name": "attribute_name",
            "description": "detailed description",
            "purpose": "business purpose"
        }}
    ]
}}

Attributes to analyze:
"""
        for attr_name, samples in attributes:
            prompt += f"\nAttribute: {attr_name}\nSamples: {samples}\n"
            
        return prompt

    def process_batch(self, batch: List[Tuple[str, str]], company_context: str) -> Dict:
        """
        Process a batch of attributes using the LLM.
        
        Args:
            batch: List of (attribute_name, sample_values) tuples
            company_context: Company-specific context
            
        Returns:
            Dictionary containing processed descriptions
        """
        try:
            prompt = self.create_prompt(batch, company_context)
            response = chinou_response(prompt)
            
            # Parse the JSON response
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return {"attributes": []}

    def process_csv(self, filename: str, company_context: str = ""):
        """
        Process a CSV file to generate descriptions for all attributes.
        
        Args:
            filename: Name of the CSV file
            company_context: Company-specific context
        """
        try:
            input_path = os.path.join(self.input_folder, filename)
            df = pd.read_csv(input_path)
            logger.info(f"Processing file: {filename} with {len(df.columns)} attributes")
            
            # Prepare attributes with their samples
            attributes = []
            for column in df.columns:
                samples = self.get_sample_values(df, column)
                attributes.append((column, samples))
            
            # Process attributes in batches
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for i in range(0, len(attributes), self.batch_size):
                    batch = attributes[i:i + self.batch_size]
                    future = executor.submit(self.process_batch, batch, company_context)
                    futures.append(future)
                
                for future in as_completed(futures):
                    result = future.result()
                    if result and "attributes" in result:
                        results.extend(result["attributes"])
            
            # Create output DataFrame
            output_df = pd.DataFrame(results)
            
            # Save results
            output_path = os.path.join(self.output_folder, filename)
            output_df.to_csv(output_path, index=False)
            logger.info(f"Successfully processed and saved descriptions to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise

def main():
    # Example usage
    generator = AttributeDescriptionGenerator(
        batch_size=3,  # Process 3 attributes per LLM call
        max_workers=4  # Use 4 parallel threads
    )
    
    company_context = """
    Our company is a financial technology firm specializing in payment processing 
    and risk management. Our data includes customer transactions, risk scores, 
    and compliance-related information.
    """
    
    # Process all CSV files in input folder
    for filename in os.listdir(generator.input_folder):
        if filename.endswith('.csv'):
            logger.info(f"Starting processing of {filename}")
            generator.process_csv(filename, company_context)
            logger.info(f"Completed processing of {filename}")

if __name__ == "__main__":
    main()
