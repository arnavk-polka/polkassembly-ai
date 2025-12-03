import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .governance.query2sql import Query2SQL
from .voting.vote_query2sql import VoteQuery2SQL

__all__ = ['Query2SQL', 'VoteQuery2SQL']

def main():
    """Example usage and testing"""
    try:
        query_processor = Query2SQL()
        
        if not query_processor.test_connection():
            print("❌ Database connection failed!")
            return
        
        print("✅ Database connection successful!")
        print(f"📊 Table: {query_processor.table_name}")
        
        example_queries = [
            "Show me the 10 most recent proposals",
            "How many Kusama proposals are there?",
            "What treasury proposals exist?",
            "Find proposals created in 2024",
            "Show me active referendums"
        ]
        
        
        for i, query in enumerate(example_queries, 1):
            
            result = query_processor.process_query(query)
            
            if result["success"]:
                print(f"✅ SQL: {result['sql_query']}")
                print(f"📊 Results: {result['result_count']} rows")
                print(f"💬 Response: {result['natural_response'][:200]}...")
            else:
                print(f"❌ Error: {result['error']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
