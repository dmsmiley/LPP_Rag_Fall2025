from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from backend.database import ChroniclesDatabase

class ChroniclesAgent:
    def __init__(self, db: ChroniclesDatabase, model_name: str, max_iter: int):
        self.db = db
        self.model_name = model_name
        self.max_iter = max_iter
        self.last_sources = []

    def create_tool(self):
        @tool("Query Chronicles Database")
        def query_chronicles_db(query: str) -> str:
            """Search the Chronicles database containing Bible Project video content and biblical text.
            
            Args:
                query: Search query about Chronicles
                
            Returns:
                Relevant passages from the database
            """
            try:
                results = self.db.query(query)
                
                if results:
                    # Store sources for UI display
                    self.last_sources.extend(results)
                    
                    passages = [row["text"] for row in results]
                    return "\n\n---\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(passages)])
                else:
                    return "No relevant passages found."
                    
            except Exception as e:
                return f"Error querying database: {str(e)}"
        
        return query_chronicles_db

    def ask(self, question: str) -> dict:
        """
        Ask a question to the agent.
        
        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        # Reset sources for this query
        self.last_sources = []
        
        llm = LLM(model=self.model_name)
        query_tool = self.create_tool()
        
        agent = Agent(
            role='Chronicles Content Assistant',
            goal='Answer questions about Chronicles using the database',
            backstory='You are an expert who has access to a database with content about the book of Chronicles.',
            tools=[query_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=self.max_iter
        )
        
        task = Task(
            description=question,
            agent=agent,
            expected_output='A comprehensive answer based on the database content'
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            max_rpm=20
        )
        
        result = crew.kickoff()
        
        return {
            "answer": str(result),
            "sources": self.last_sources.copy()
        }
