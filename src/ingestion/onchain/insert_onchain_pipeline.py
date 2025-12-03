#!/usr/bin/env python3
"""
Comprehensive script to download all onchain data, process it through the pipeline,
and identify new records not yet in the PostgreSQL database.

Pipeline:
1. Download onchain data (JSON format) using onchain_data.py
2. Flatten JSON to CSV files using flatten_all_data.py
3. Combine all CSV files into one table using create_one_table.py
4. Filter to 86 columns using filter_data.py
5. Compare with existing PostgreSQL data and identify new records
6. Save new records to CSV for insertion

All processed data is stored in updates/data folder.
"""

import os
import sys
import pandas as pd
import psycopg2
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging
from datetime import datetime
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnchainDataPipeline:
    def __init__(self, updates_data_dir: str = None):
        """Initialize the onchain data pipeline"""
        self.updates_dir = Path(__file__).parent
        self.updates_data_dir = Path(updates_data_dir) if updates_data_dir else self.updates_dir / "data"
        
        self.raw_data_dir = self.updates_data_dir / "raw_json"
        self.csv_data_dir = self.updates_data_dir / "all_csv"
        self.combined_data_dir = self.updates_data_dir / "one_table"
        self.filtered_data_dir = self.updates_data_dir / "filtered"
        self.new_records_dir = self.updates_data_dir / "new_records"
        
        for directory in [self.raw_data_dir, self.csv_data_dir, self.combined_data_dir, 
                         self.filtered_data_dir, self.new_records_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized with data directory: {self.updates_data_dir}")
        
        self.filtered_columns = [
            'source_file', 'source_network', 'source_proposal_type',
            'source_row_id', 'allowedcommentor', 'content', 'createdat',
            'datasource', 'hash', 'history_0_content',
            'history_0_createdat_nanoseconds', 'history_0_createdat_seconds',
            'history_0_title', 'history_1_content',
            'history_1_createdat_nanoseconds', 'history_1_createdat_seconds',
            'history_1_title', 'history_2_content',
            'history_2_createdat_nanoseconds', 'history_2_createdat_seconds',
            'history_2_title', 'id', 'index', 'isdefaultcontent', 'isdeleted',
            'linkedpost_indexorhash', 'linkedpost_proposaltype', 'metrics_comments',
            'metrics_reactions_dislike', 'metrics_reactions_like', 'network',
            'onchaininfo_beneficiaries_0_address',
            'onchaininfo_beneficiaries_0_amount',
            'onchaininfo_beneficiaries_0_assetid', 'onchaininfo_createdat',
            'onchaininfo_curator', 'onchaininfo_decisionperiodendsat',
            'onchaininfo_description', 'onchaininfo_hash', 'onchaininfo_index',
            'onchaininfo_origin', 'onchaininfo_prepareperiodendsat',
            'onchaininfo_proposer', 'onchaininfo_reward', 'onchaininfo_status',
            'onchaininfo_type', 'onchaininfo_votemetrics',
            'onchaininfo_votemetrics_aye_count',
            'onchaininfo_votemetrics_aye_value',
            'onchaininfo_votemetrics_bareayes_value',
            'onchaininfo_votemetrics_nay_count',
            'onchaininfo_votemetrics_nay_value',
            'onchaininfo_votemetrics_support_value', 'poll', 'proposaltype',
            'publicuser_addresses_0', 'publicuser_addresses_1',
            'publicuser_addresses_2', 'publicuser_addresses_3',
            'publicuser_addresses_4', 'publicuser_createdat', 'publicuser_id',
            'publicuser_profiledetails_bio', 'publicuser_profiledetails_coverimage',
            'publicuser_profiledetails_image',
            'publicuser_profiledetails_publicsociallinks_0_platform',
            'publicuser_profiledetails_publicsociallinks_0_url',
            'publicuser_profiledetails_title', 'publicuser_profilescore',
            'publicuser_rank', 'publicuser_username', 'tags_0_lastusedat',
            'tags_0_network', 'tags_0_value', 'tags_1_lastusedat', 'tags_1_network',
            'tags_1_value', 'tags_2_lastusedat', 'tags_2_network', 'tags_2_value',
            'title', 'topic', 'updatedat', 'userid', 'row_index'
        ]

    def step1_download_onchain_data(self) -> bool:
        """Step 1: Download all onchain data in JSON format"""
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading onchain data (JSON format)")
        logger.info("=" * 60)
        
        try:
            data_dir = project_root / "src" / "data"
            onchain_script = data_dir / "onchain_data.py"
            
            if not onchain_script.exists():
                logger.error(f"onchain_data.py not found at {onchain_script}")
                return False
            
            logger.info(f"Running onchain_data.py with custom data directory: {self.raw_data_dir}")
            
            custom_script_content = f'''
import sys
sys.path.insert(0, "{project_root / "src"}")

from src.ingestion.onchain.onchain_data import fetch_onchain_data

print("Starting onchain data fetch...")
fetch_onchain_data(max_items_per_type=1000, data_dir="{self.raw_data_dir}")
print("Onchain data fetch completed!")
'''
            
            custom_script_path = self.updates_data_dir / "temp_onchain.py"
            with open(custom_script_path, 'w') as f:
                f.write(custom_script_content)
            
            logger.info("Starting onchain_data.py execution...")
            process = subprocess.Popen([
                sys.executable, str(custom_script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"[onchain_data.py] {output.strip()}")
            
            return_code = process.poll()
            
            custom_script_path.unlink()
            
            if return_code != 0:
                logger.error(f"onchain_data.py failed with return code {return_code}")
                return False
            
            logger.info("onchain_data.py completed successfully")
            
            json_files = list(self.raw_data_dir.glob("*.json"))
            logger.info(f"Downloaded {len(json_files)} JSON files")
            
            if len(json_files) == 0:
                logger.error("No JSON files were downloaded!")
                return False
            
            total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)
            logger.info(f"Total data size: {total_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in step 1 (download onchain data): {e}")
            return False

    def step2_flatten_json_to_csv(self) -> bool:
        """Step 2: Flatten all JSON files to CSV format"""
        logger.info("=" * 60)
        logger.info("STEP 2: Flattening JSON to CSV files")
        logger.info("=" * 60)
        
        try:
            texttosql_dir = project_root / "src" / "texttosql"
            flatten_script = texttosql_dir / "flatten_all_data.py"
            
            if not flatten_script.exists():
                logger.error(f"flatten_all_data.py not found at {flatten_script}")
                return False
            
            logger.info(f"Running flatten_all_data.py from {texttosql_dir}")
            
            custom_script_content = f'''
import sys
sys.path.insert(0, "{project_root / "src"}")

from src.ingestion.onchain.flatten_all_data import DataFlattener

flattener = DataFlattener(
    data_dir="{self.raw_data_dir}",
    output_data_dir="{self.updates_data_dir}"
)

print("Starting JSON flattening process...")
flattener.process_all_files()
print("JSON flattening completed!")
'''
            
            custom_script_path = self.updates_data_dir / "temp_flatten.py"
            with open(custom_script_path, 'w') as f:
                f.write(custom_script_content)
            
            logger.info("Starting flatten_all_data.py execution...")
            process = subprocess.Popen([
                sys.executable, str(custom_script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"[flatten_all_data.py] {output.strip()}")
            
            return_code = process.poll()
            
            custom_script_path.unlink()
            
            if return_code != 0:
                logger.error(f"flatten_all_data.py failed with return code {return_code}")
                return False
            
            logger.info("flatten_all_data.py completed successfully")
            
            csv_files = list(self.csv_data_dir.glob("*.csv"))
            logger.info(f"Created {len(csv_files)} CSV files")
            
            if len(csv_files) == 0:
                logger.error("No CSV files were created!")
                return False
            
            for csv_file in csv_files[:5]:
                df = pd.read_csv(csv_file)
                logger.info(f"  {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
            
            if len(csv_files) > 5:
                logger.info(f"  ... and {len(csv_files) - 5} more files")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in step 2 (flatten JSON to CSV): {e}")
            return False

    def step3_combine_csv_files(self) -> bool:
        """Step 3: Combine all CSV files into one table (486 columns)"""
        logger.info("=" * 60)
        logger.info("STEP 3: Combining CSV files into one table")
        logger.info("=" * 60)
        
        try:
            custom_script_content = f'''
import sys
sys.path.insert(0, "{project_root / "src"}")

from src.ingestion.onchain.create_one_table import CSVCombiner

combiner = CSVCombiner(
    csv_directory="{self.csv_data_dir}",
    output_directory="{self.combined_data_dir}"
)

print("Starting CSV combination process...")
result = combiner.run_combination()
combined_file = result['combined_csv']
print(f"Combined file created: {{combined_file}}")
'''
            
            custom_script_path = self.updates_data_dir / "temp_combine.py"
            with open(custom_script_path, 'w') as f:
                f.write(custom_script_content)
            
            logger.info("Starting create_one_table.py execution...")
            process = subprocess.Popen([
                sys.executable, str(custom_script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"[create_one_table.py] {output.strip()}")
            
            return_code = process.poll()
            
            custom_script_path.unlink()
            
            if return_code != 0:
                logger.error(f"create_one_table.py failed with return code {return_code}")
                return False
            
            logger.info("create_one_table.py completed successfully")
            
            combined_files = list(self.combined_data_dir.glob("combined_*.csv"))
            if not combined_files:
                logger.error("Combined CSV file was not created!")
                return False
            
            combined_file = combined_files[0]
            
            df_combined = pd.read_csv(combined_file)
            logger.info(f"Combined file created: {combined_file.name}")
            logger.info(f"  Rows: {len(df_combined):,}")
            logger.info(f"  Columns: {len(df_combined.columns)} (expected ~486)")
            logger.info(f"  File size: {combined_file.stat().st_size / (1024*1024):.2f} MB")
            
            self.combined_file_path = combined_file
            
            return True
            
        except Exception as e:
            logger.error(f"Error in step 3 (combine CSV files): {e}")
            return False

    def step4_filter_to_86_columns(self) -> bool:
        """Step 4: Filter combined data to 86 columns"""
        logger.info("=" * 60)
        logger.info("STEP 4: Filtering to 86 columns")
        logger.info("=" * 60)
        
        try:
            if not hasattr(self, 'combined_file_path'):
                combined_files = list(self.combined_data_dir.glob("combined_*.csv"))
                if not combined_files:
                    logger.error("No combined CSV file found!")
                    return False
                self.combined_file_path = combined_files[0]
            
            logger.info(f"Loading combined file: {Path(self.combined_file_path).name}")
            df_combined = pd.read_csv(self.combined_file_path)
            
            logger.info(f"Original data: {len(df_combined)} rows, {len(df_combined.columns)} columns")
            
            available_columns = set(df_combined.columns)
            requested_columns = set(self.filtered_columns)
            
            missing_columns = requested_columns - available_columns
            extra_columns = available_columns - requested_columns
            
            if missing_columns:
                logger.warning(f"Missing columns ({len(missing_columns)}): {list(missing_columns)[:10]}...")
            
            columns_to_use = [col for col in self.filtered_columns if col in available_columns]
            logger.info(f"Using {len(columns_to_use)} out of {len(self.filtered_columns)} requested columns")
            
            df_filtered = df_combined[columns_to_use]
            
            filtered_file_path = self.filtered_data_dir / "governance_data_86.csv"
            df_filtered.to_csv(filtered_file_path, index=False)
            
            logger.info(f"Filtered data saved: {filtered_file_path.name}")
            logger.info(f"  Rows: {len(df_filtered):,}")
            logger.info(f"  Columns: {len(df_filtered.columns)}")
            logger.info(f"  File size: {filtered_file_path.stat().st_size / (1024*1024):.2f} MB")
            
            self.filtered_file_path = filtered_file_path
            
            return True
            
        except Exception as e:
            logger.error(f"Error in step 4 (filter to 86 columns): {e}")
            return False

    def step5_identify_new_records(self) -> bool:
        """Step 5: Compare with PostgreSQL and identify new records"""
        logger.info("=" * 60)
        logger.info("STEP 5: Identifying new records not in PostgreSQL")
        logger.info("=" * 60)
        
        try:
            if not hasattr(self, 'filtered_file_path'):
                filtered_files = list(self.filtered_data_dir.glob("governance_data_86.csv"))
                if not filtered_files:
                    logger.error("No filtered CSV file found!")
                    return False
                self.filtered_file_path = filtered_files[0]
            
            logger.info(f"Loading filtered data: {self.filtered_file_path.name}")
            df_new = pd.read_csv(self.filtered_file_path)
            
            if 'row_index' not in df_new.columns:
                logger.error("row_index column not found in filtered data!")
                return False
            
            csv_row_indexes = set(df_new['row_index'].dropna().astype(str))
            logger.info(f"Found {len(csv_row_indexes)} row_index values in CSV")
            
            try:
                db_config = {
                    'host': os.getenv('POSTGRES_HOST'),
                    'port': os.getenv('POSTGRES_PORT', '5432'),
                    'database': os.getenv('POSTGRES_DATABASE'),
                    'user': os.getenv('POSTGRES_USER'),
                    'password': os.getenv('POSTGRES_PASSWORD')
                }
                
                required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
                missing_vars = [var for var in required_vars if not os.getenv(var)]
                
                if missing_vars:
                    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                    return False
                
                logger.info("Connecting to PostgreSQL database...")
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'governance_data'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    logger.warning("governance_data table does not exist in PostgreSQL")
                    logger.info("All records will be considered new")
                    db_row_indexes = set()
                else:
                    logger.info("Querying existing row_index values from governance_data table...")
                    cursor.execute("SELECT row_index FROM governance_data WHERE row_index IS NOT NULL")
                    db_results = cursor.fetchall()
                    db_row_indexes = set(str(row[0]) for row in db_results)
                    logger.info(f"Found {len(db_row_indexes)} existing row_index values in database")
                
                cursor.close()
                conn.close()
                
            except Exception as db_error:
                logger.error(f"Database connection error: {db_error}")
                logger.info("Assuming no existing records in database")
                db_row_indexes = set()
            
            new_row_indexes = csv_row_indexes - db_row_indexes
            logger.info(f"Found {len(new_row_indexes)} new records to insert")
            
            if len(new_row_indexes) == 0:
                logger.info("No new records found. Database is up to date.")
                return True
            
            df_new_records = df_new[df_new['row_index'].astype(str).isin(new_row_indexes)]
            
            new_records_file = self.new_records_dir / f"new_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_new_records.to_csv(new_records_file, index=False)
            
            logger.info(f"New records saved: {new_records_file.name}")
            logger.info(f"  Records: {len(df_new_records):,}")
            logger.info(f"  Columns: {len(df_new_records.columns)}")
            logger.info(f"  File size: {new_records_file.stat().st_size / (1024*1024):.2f} MB")
            
            if 'createdat' in df_new_records.columns:
                summary_data = df_new_records[['row_index', 'createdat']].copy()
                summary_file = self.new_records_dir / f"new_records_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                summary_data.to_csv(summary_file, index=False)
                logger.info(f"Summary file saved: {summary_file.name}")
            
            self.new_records_file_path = new_records_file
            
            return True
            
        except Exception as e:
            logger.error(f"Error in step 5 (identify new records): {e}")
            return False

    def run_full_pipeline(self) -> bool:
        """Run the complete onchain data pipeline"""
        logger.info("🚀 Starting Onchain Data Pipeline")
        logger.info(f"📁 Data directory: {self.updates_data_dir}")
        logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        
        if not self.step1_download_onchain_data():
            logger.error("❌ Pipeline failed at step 1")
            return False
        
        if not self.step2_flatten_json_to_csv():
            logger.error("❌ Pipeline failed at step 2")
            return False
        
        if not self.step3_combine_csv_files():
            logger.error("❌ Pipeline failed at step 3")
            return False
        
        if not self.step4_filter_to_86_columns():
            logger.error("❌ Pipeline failed at step 4")
            return False
        
        if not self.step5_identify_new_records():
            logger.error("❌ Pipeline failed at step 5")
            return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total duration: {duration}")
        logger.info(f"📁 All data saved in: {self.updates_data_dir}")
        
        if hasattr(self, 'new_records_file_path'):
            logger.info(f"📄 New records file: {self.new_records_file_path.name}")
        
        return True

    def cleanup_intermediate_files(self, keep_final_results: bool = True):
        """Clean up intermediate files to save space"""
        logger.info("🧹 Cleaning up intermediate files...")
        
        if not keep_final_results:
            logger.warning("This will delete ALL processed data!")
        
        json_files = list(self.raw_data_dir.glob("*.json"))
        if json_files:
            total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)
            logger.info(f"Raw JSON files: {len(json_files)} files, {total_size:.2f} MB")
            

def main():
    """Main execution function"""
    logger.info("🔄 Onchain Data Pipeline - Insert to PostgreSQL")
    
    pipeline = OnchainDataPipeline()
    
    success = pipeline.run_full_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        logger.info("📋 Next steps:")
        logger.info("   1. Review the new records CSV file")
        logger.info("   2. Use insert_into_postgres.py to insert new records")
        logger.info("   3. Verify the insertion was successful")
    else:
        logger.error("❌ Pipeline failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
