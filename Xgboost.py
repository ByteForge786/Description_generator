import pandas as pd
import concurrent.futures
from typing import List, Dict
import time
from test import chinou_response
import queue
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

@dataclass
class ProcessingConfig:
    batch_size: int = 5  # Number of items to process in each batch
    max_workers: int = 4  # Number of parallel threads
    timeout_seconds: int = 30  # Timeout for each batch

class DescriptionGenerator:
    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
        self.result_queue = queue.Queue()
        self.error_queue = queue.Queue()
    
    def generate_prompt(self, names: List[str]) -> str:
        """Generate a prompt for batch processing"""
        prompt = """Please analyze the following financial holding names and provide detailed descriptions.
        Format your response as a JSON array with objects containing 'name' and 'description' fields.
        Names to analyze: {names}
        
        Expected format:
        [
            {{"name": "holding_name", "description": "detailed_description"}},
            ...
        ]
        """.format(names=names)
        return prompt

    def parse_response(self, response: str) -> List[Dict[str, str]]:
        """Parse the LLM response into structured data"""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}")

    def process_batch(self, names: List[str]) -> List[Dict[str, str]]:
        """Process a batch of names"""
        try:
            prompt = self.generate_prompt(names)
            response = chinou_response(prompt)
            return self.parse_response(response)
        except Exception as e:
            self.error_queue.put((names, str(e)))
            return []

    def process_in_parallel(self, df: pd.DataFrame, name_column: str) -> pd.DataFrame:
        """Process the entire dataset using parallel threads and batching"""
        names = df[name_column].tolist()
        batches = [
            names[i:i + self.config.batch_size] 
            for i in range(0, len(names), self.config.batch_size)
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_batch, batch): batch 
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.config.timeout_seconds)
                    results.extend(batch_results)
                except Exception as e:
                    self.error_queue.put((batch, str(e)))

        # Create results DataFrame
        results_df = pd.DataFrame(results)
        # Merge with original DataFrame
        return df.merge(results_df, left_on=name_column, right_on='name', how='left')

def process_csv(
    input_file: str,
    output_file: str,
    name_column: str = 'name',
    config: ProcessingConfig = ProcessingConfig()
) -> None:
    """Main function to process CSV file"""
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Initialize generator
    generator = DescriptionGenerator(config)
    
    # Process data
    start_time = time.time()
    result_df = generator.process_in_parallel(df, name_column)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Print processing summary
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Print any errors
    while not generator.error_queue.empty():
        batch, error = generator.error_queue.get()
        print(f"Error processing batch {batch}: {error}")

if __name__ == "__main__":
    # Example usage
    config = ProcessingConfig(
        batch_size=5,  # Process 5 names at a time
        max_workers=4,  # Use 4 parallel threads
        timeout_seconds=30  # 30 second timeout per batch
    )
    
    process_csv(
        input_file="holdings.csv",
        output_file="holdings_with_descriptions.csv",
        name_column="holding_name",
        config=config
    )
