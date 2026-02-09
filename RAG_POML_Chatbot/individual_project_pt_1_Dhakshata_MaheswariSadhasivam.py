"""
Prompt Engineering Project
UNCC - Design and Development of Generative AI Applications

Name: Dhakshata Maheswari Sadhasivam

This script runs a command line chatbot that compares responses from different LLM models.
It implements different prompting techniques and accesses LLMs through the Groq API.
"""

# =======================  Installation Instructions  ========================
# Before running this code, create a virtual environment that you can use for class work.
# Then with the venv activated, run the following command to install the required packages:
# pip install groq python-dotenv,matplotlib
# You will then need to make a .env file that has GROQ_API_KEY set to your api key value. 
# ============================================================================

import os
from groq import Groq
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

load_dotenv()  # Loads variables from .env

# Make sure you have your Groq API key saved in a .env file as GROQ_API_KEY 
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# ============================================================================
# Groq Client Wrapper
# ============================================================================

class GroqClient:
    """Wrapper class for Groq API interactions"""
    
    def __init__(self, api_key):
        """
        Initialize the Groq client
        
        Args:
            api_key (str): Groq API key
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.client = Groq(api_key=api_key)
        
    
    def call_llm(self, model_name, messages):
        """
        Query the Groq API with a list of messages
        
        Args:
            model_name (str): The name of the model to use
            messages (list): List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: The model's response or an error message
        """
        try:
            completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5
        )
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error querying the LLM: {e}"


# ============================================================================
# Chatbot class
# ============================================================================

class PythonHelpBot:
    """Command-line chatbot that compares LLM responses using different prompting techniques"""
    
    # TODO: Define the models to use for comparison
    # MODEL_A, MODEL_B, MODEL_C, MODEL_D should be the model names
    # EVALUATOR_MODEL should be the model used to evaluate responses
    MODEL_A = "meta-llama/llama-4-scout-17b-16e-instruct"
    MODEL_B = "openai/gpt-oss-120b"
    MODEL_C = "groq/compound"
    MODEL_D = "qwen/qwen3-32b"
    EVALUATOR_MODEL = "llama-3.3-70b-versatile"
    #EVALUATOR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    MODEL_SIZES = {
        "A": 17,   # 17B (Llama-4-scout)
        "B": 120,  # 120B (GPT-OSS)
        "C": 70,   # 70B (Compound/Llama)
        "D": 32    # 32B (Qwen)
    }

    
    # Prompting technique options
    PROMPT_TECHNIQUES = {
        "1": "Zero-Shot",
        "2": "Few-Shot",
        "3": "Chain-of-Thought"
        # plus any others you want to add
    }

    
    
    def __init__(self, groq_client):
        """
        Initialize the chatbot
        
        Args:
            groq_client (GroqClient): Instance of GroqClient for API interactions
        """
        self.groq_client = groq_client
        
        # System prompt (added at the start of every conversation)
        # TODO: Customize this system prompt for the Python Help Assistant
        self.system_prompt = """ You are a helpful Python programming assistant.Your goal is to provide correct, clear, and beginner-friendly explanations.
Follow Python best practices and include short code examples when helpful.Limit explanations to one sentence maximum. Do not use conversational filler."""
        
        # TODO: Define your prompt templates here for the Python Help Assistant
        # Each prompt should include a {user_query} placeholder that will be replaced with the user's question at runtime
        self.prompts = {
            "Zero-Shot": """Answer the following Python programming question clearly and correctly.
User query: {user_query}""",
            
            "Few-Shot": """You are a Python expert.Limit explanations to one sentence maximum. Do not use conversational filler.

            Example 1:
            Q: How do I print text in Python?
            A: You can use the print() function. Example:
            print("Hello World")

            Example 2:
            Q: How do I create a list?
            A: Lists are created using square brackets. Example:
            numbers = [1, 2, 3]

            Now answer this question:
User query: {user_query}""",
            
            "Chain-of-Thought": """You are a Python expert.Limit explanations to one sentence maximum. Do not use conversational filler.
            Think step-by-step to solve the problem internally, then provide a clear final answer.
            Do NOT show your reasoning steps.

User query: {user_query}"""
        }
        # Track how often each model is selected as best
        self.win_counts = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0
        }

    
    def display_welcome(self):
        """Display welcome message"""
        print("\nWelcome to the Python Help Assistant!")
        print("\nThis chatbot will compare responses from four different models:")
        print(f"  Model A: {self.MODEL_A}")
        print(f"  Model B: {self.MODEL_B}")
        print(f"  Model C: {self.MODEL_C}")
        print(f"  Model D: {self.MODEL_D}")
        print(f"  The evaluator model, {self.EVALUATOR_MODEL}, will assess which response is better.")
    
    def display_prompt_techniques(self):
        """Display available prompting techniques"""
        print("\nChoose a prompting technique:")
        for key, technique in self.PROMPT_TECHNIQUES.items():
            print(f"{key}. {technique}")
        print("4. Exit")
    
    def get_user_input(self, prompt):
        """
        Get user input
        
        Args:
            prompt (str): Prompt to display to user
            
        Returns:
            str: User's input
        """
        return input(prompt).strip()
    
    def select_prompt_technique(self):
        """
        Handle prompt technique selection
        
        Returns:
            str: Selected prompt technique name, or None if invalid/exit
        """
        self.display_prompt_techniques()
        choice = self.get_user_input("Enter your choice (1, 2, 3, or 4 to exit): ")
        
        if choice == '4':
            return 'exit'
        
        if choice in self.PROMPT_TECHNIQUES:
            return self.PROMPT_TECHNIQUES[choice]
        else:
            print("Invalid choice! Please select 1, 2, 3, or 4.")
            return None
    
    def get_user_query(self):
        """
        Get the user's query
        
        Returns:
            str: User's query
        """
        print("\n" + "="*60)
        print("Please enter your Python programming question.")
        # TODO: Add an example question relevant to Python
        print("Example: How do I read a file line by line in Python?")
        print("="*60)
        user_query = self.get_user_input("\nYour query: ")
        
        if not user_query:
            user_query = "How do I print hello world?"
        
        return user_query
    
    def query_model(self, model_name, prompt_type, user_query):
        """
        Query a single model with the user's question
        
        Args:
            model_name (str): Name of the model to query
            prompt_type (str): Type of prompt technique to use
            user_query (str): User's movie query
            
        Returns:
            str: Model's response
        """
        # Format the technique-specific prompt with the user's query
        user_message = self.prompts[prompt_type].format(user_query=user_query).strip()
        
        # Create messages list with system prompt and user message
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        start_time = time.time()

        
        # Call the model
        response = self.groq_client.call_llm(model_name, messages)

        end_time = time.time()    # End Timer
        duration = round(end_time - start_time, 2)
        return response,duration
    
    def evaluate_responses(self, response_a, response_b, response_c, response_d):
        """
        Use evaluator model to compare four responses
        
        Args:
            response_a (str): Response from Model A
            response_b (str): Response from Model B
            response_c (str): Response from Model C
            response_d (str): Response from Model D
            
        Returns:
            str: Evaluation result
        """
        # TODO: Create an evaluation prompt that compares the four responses
        # Your prompt should:
        # 1. Explain that you're comparing four responses to a Python question
        # 2. List criteria to consider (correctness, clarity, best practices)
        # 3. Include all four responses (A, B, C, D)
        # 4. Ask for a determination of which is better and why

        
        evaluation_prompt = """
You are comparing four responses to the same Python programming question.

Evaluate each response based on:
1. Correctness
2. Clarity
3. Use of Python best practices
4. Helpfulness for a learner

After reviewing all responses, decide which one is the best overall.

Respond in the following format:
- Best Response: A / B / C / D
- Explanation: Short justification

Response A:
{response_a}

Response B:
{response_b}

Response C:
{response_c}

Response D:
{response_d}
""".format(response_a=response_a, response_b=response_b, response_c=response_c, response_d=response_d)
        
        messages = [
            {"role": "system", "content": "You are an expert evaluator comparing responses."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        return self.groq_client.call_llm(self.EVALUATOR_MODEL, messages)
   
   
    def extract_winner(self, evaluation_text):
        """Extract winning model from evaluator output like:
        '- Best Response: C'"""
        evaluation_text = evaluation_text.strip().lower()

        if "best response" not in evaluation_text:
            return None

        # Look specifically for A/B/C/D after colon
        for model in ["a", "b", "c", "d"]:
            if f": {model}" in evaluation_text or f":{model}" in evaluation_text:
                return model.upper()

        return None
    
    def plot_performance(self, session_data, complexity_label):
        """Generates a 3-panel graph for the current session"""
        if not session_data:
            print("No data to plot.")
            return

        print(f"Generating graph for {complexity_label} complexity...")
        
        technique_names = list(self.PROMPT_TECHNIQUES.values())
        models = ["A", "B", "C", "D"]
        
      
        colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B'] 
        
        
        plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, facecolor='#f8f9fa')
        fig.suptitle(f"Model Latency by Technique ({complexity_label} Complexity)", 
                     fontsize=18, fontweight='bold', color='#333333', y=0.98)

 
        for i, ax in enumerate(axes):
        
            if i < len(session_data):
                data = session_data[i]
                times = [data["times"][m] for m in models]
                winner = data["winner"]
                tech_name = technique_names[i]
            else:
                continue

    
            ax.set_facecolor('#ffffff') 
            ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
            
            # --- PLOT: Bars ---
            bars = ax.bar(models, times, color=colors, zorder=3, width=0.6, alpha=0.9)
            
         
            if winner in models:
                idx = models.index(winner)
                #
                ax.text(idx, times[idx] + (max(times)*0.05), "★ WIN", 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='#D0021B')
                #
                bars[idx].set_edgecolor('#333333')
                bars[idx].set_linewidth(1.5)

           
            for bar, t in zip(bars, times):
                height = bar.get_height()
                #
                ax.text(bar.get_x() + bar.get_width()/2., height - (height*0.1) if height > 0.1 else height + 0.01,
                        f'{t:.2f}s', ha='center', va='bottom' if height < 0.1 else 'top', 
                        color='white' if height > 0.1 else 'black', fontweight='bold', fontsize=9)

           
            ax.set_title(tech_name, fontsize=14, pad=15, color='#444444', fontweight='medium')
            
         
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
    
            labels = [f"{m}\n{self.MODEL_SIZES[m]}B" for m in models]
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(labels, color='#555555')

            if i == 0:
                ax.set_ylabel("Response Time (Seconds)", fontsize=12, color='#555555')

        handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
        fig.legend(handles, [f"Model {m} ({self.MODEL_SIZES[m]}B)" for m in models], 
                   loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False, fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.88]) 
        print("Displaying graph... (Close window to exit)")
        plt.show()



    def run(self):
        """Main chatbot execution"""
        self.display_welcome()

        # Step 1: collect one query per prompt technique
        technique_queries = {}
        # Storage for this specific run
        session_data = []

        print("\nEnter ONE question for EACH prompting technique:\n")

        for key, technique in self.PROMPT_TECHNIQUES.items():
            query = input(f"{technique} query: ").strip()
            if not query:
                query = "How do I print hello world in Python?"
            technique_queries[technique] = query

        # Step 2: evaluate each prompt technique independently
        for technique, user_query in technique_queries.items():
            print("\n" + "=" * 80)
            print(f"PROMPT TECHNIQUE: {technique}")
            print(f"QUESTION: {user_query}")
            print("=" * 80)

            # Query all models
            resp_a, time_a = self.query_model(self.MODEL_A, technique, user_query)
            resp_b, time_b = self.query_model(self.MODEL_B, technique, user_query)
            resp_c, time_c = self.query_model(self.MODEL_C, technique, user_query)
            resp_d, time_d = self.query_model(self.MODEL_D, technique, user_query)


            print("\n" + "=" * 80)
            print(f"MODEL A ({self.MODEL_A})")
            print("=" * 80)
            print(resp_a)

            print("\n" + "=" * 80)
            print(f"MODEL B ({self.MODEL_B})")
            print("=" * 80)
            print(resp_b)

            print("\n" + "=" * 80)
            print(f"MODEL C ({self.MODEL_C})")
            print("=" * 80)
            print(resp_c)

            print("\n" + "=" * 80)
            print(f"MODEL D ({self.MODEL_D})")
            print("=" * 80)
            print(resp_d)


            # Evaluate
            evaluation = self.evaluate_responses(
                resp_a, resp_b, resp_c, resp_d
            )

            print("\nEVALUATION RESULT:")
            print(evaluation)

            # Update win count
            winner = self.extract_winner(evaluation)
            if winner:
                self.win_counts[winner] += 1

            session_data.append({
                "times": {"A": time_a, "B": time_b, "C": time_c, "D": time_d},
                "winner": winner
            })

        print("\n" + "=" * 80)
        complexity_label = input("Enter the Complexity Label for this run (Low/Medium/High): ").strip()
        if not complexity_label: complexity_label = "Unspecified"
        
        
        # Step 3: final aggregated result
        print("\n" + "=" * 80)
        print("FINAL WIN FREQUENCY (Across ALL Prompt Techniques)")
        print("=" * 80)

        for model, count in self.win_counts.items():
            print(f"Model {model}: {count}")
        
        
        self.plot_performance(session_data, complexity_label)

        print("\nGoodbye!\n")



# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Initialize Groq client
    groq_client = GroqClient(GROQ_API_KEY)
    
    # Initialize and run chatbot
    chatbot = PythonHelpBot(groq_client)
    chatbot.run()
