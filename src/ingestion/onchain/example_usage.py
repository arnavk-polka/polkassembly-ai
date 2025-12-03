#!/usr/bin/env python3
"""
Example usage of the onchain data pipeline.
This script shows different ways to use the pipeline.
"""

import os
import sys
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from insert_onchain_to_postgres import OnchainDataPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_full_pipeline():
    """Example: Run the complete pipeline"""
    logger.info("Example 1: Running complete pipeline")
    
    pipeline = OnchainDataPipeline()
    
    success = pipeline.run_full_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        return True
    else:
        logger.error("❌ Pipeline failed!")
        return False

def example_custom_data_directory():
    """Example: Run pipeline with custom data directory"""
    logger.info("Example 2: Running pipeline with custom data directory")
    
    custom_data_dir = "/tmp/polkassembly_updates"
    
    pipeline = OnchainDataPipeline(updates_data_dir=custom_data_dir)
    
    success = pipeline.run_full_pipeline()
    
    if success:
        logger.info(f"✅ Pipeline completed! Data saved in: {custom_data_dir}")
        return True
    else:
        logger.error("❌ Pipeline failed!")
        return False

def example_individual_steps():
    """Example: Run individual pipeline steps"""
    logger.info("Example 3: Running individual pipeline steps")
    
    pipeline = OnchainDataPipeline()
    
    steps = [
        ("Download onchain data", pipeline.step1_download_onchain_data),
        ("Flatten JSON to CSV", pipeline.step2_flatten_json_to_csv),
        ("Combine CSV files", pipeline.step3_combine_csv_files),
        ("Filter to 86 columns", pipeline.step4_filter_to_86_columns),
        ("Identify new records", pipeline.step5_identify_new_records)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Running: {step_name}")
        
        success = step_func()
        if not success:
            logger.error(f"❌ Step failed: {step_name}")
            return False
        
        logger.info(f"✅ Step completed: {step_name}")
        
    
    logger.info("✅ All individual steps completed!")
    return True

def example_with_error_handling():
    """Example: Pipeline with comprehensive error handling"""
    logger.info("Example 4: Pipeline with error handling")
    
    try:
        pipeline = OnchainDataPipeline()
        
        if not pipeline.updates_data_dir.exists():
            logger.error("Data directory doesn't exist!")
            return False
        
        success = pipeline.run_full_pipeline()
        
        if success:
            new_records_files = list(pipeline.new_records_dir.glob("new_records_*.csv"))
            
            if new_records_files:
                latest_file = max(new_records_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"📄 Latest new records file: {latest_file.name}")
                
            else:
                logger.info("ℹ️  No new records found")
            
            return True
        else:
            logger.error("❌ Pipeline failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def example_cleanup():
    """Example: Cleanup intermediate files after pipeline"""
    logger.info("Example 5: Pipeline with cleanup")
    
    pipeline = OnchainDataPipeline()
    
    success = pipeline.run_full_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed!")
        
        logger.info("🧹 Cleaning up intermediate files...")
        pipeline.cleanup_intermediate_files(keep_final_results=True)
        
        return True
    else:
        logger.error("❌ Pipeline failed!")
        return False

def main():
    """Run example based on command line argument"""
    examples = {
        "1": ("Full Pipeline", example_full_pipeline),
        "2": ("Custom Directory", example_custom_data_directory),
        "3": ("Individual Steps", example_individual_steps),
        "4": ("Error Handling", example_with_error_handling),
        "5": ("With Cleanup", example_cleanup)
    }
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        logger.info("🔄 Onchain Data Pipeline Examples")
        logger.info("=" * 40)
        for num, (name, _) in examples.items():
            logger.info(f"{num}. {name}")
        logger.info("=" * 40)
        
        example_num = input("Choose example (1-5): ").strip()
    
    if example_num in examples:
        example_name, example_func = examples[example_num]
        logger.info(f"🚀 Running example: {example_name}")
        
        success = example_func()
        
        if success:
            logger.info("✅ Example completed successfully!")
            return 0
        else:
            logger.error("❌ Example failed!")
            return 1
    else:
        logger.error(f"❌ Invalid example number: {example_num}")
        return 1

if __name__ == "__main__":
    exit(main())
