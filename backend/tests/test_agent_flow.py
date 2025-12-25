import os
import sys
import asyncio
import langchain

# Enable Live Tracing
langchain.debug = True

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.agent_service import AgentOrchestrator
from dotenv import load_dotenv

# Load env 
load_dotenv()
# Note: Ensure GROQ_API_KEY is in .env

async def test_agent():
    print("--- Starting Agent Test ---")
    
    agent = AgentOrchestrator()
    

    # Test 3: Complex/Graph
    query_3 = "how is physical activity related to alzheimer's disease?"
    print(f"\nQuery: {query_3}")
    response_3 = await agent.run(query_3, user_role="doctor", patient_id="p001")
    print(f"Response: {response_3}\n")

if __name__ == "__main__":
    asyncio.run(test_agent())
