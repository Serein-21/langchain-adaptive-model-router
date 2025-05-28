from os import system
from click import prompt
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
    RunnableBranch
)
from typing import List, Annotated, Optional, Literal
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ============================================================================
# MODELS CONFIGURATION
# ============================================================================

class QUESTIONSENTIMENT(BaseModel):
    """Pydantic model for question complexity classification."""

    sentiment: Literal['simple', 'complex'] = Field(
        default='simple',
        description="""Classify the question complexity for optimal AI model routing:

SIMPLE: Questions requiring basic information retrieval, simple calculations, definitions, or straightforward explanations. Examples: "What is photosynthesis?", "Calculate 15% of 200", "Translate this to Spanish"

COMPLEX: Questions requiring multi-step reasoning, mathematical proofs, strategic analysis, algorithm design, scientific problem-solving, or deep domain expertise. Examples: "Prove mathematical theorems", "Design system architecture", "Analyze business strategy", "Solve multi-variable optimization problems"""
    )

    model_names: Literal["gpt-4o-mini", "o3-mini"] = Field(
        description="The name of the model that should be chosen: 'gpt-4o-mini' for simple questions, 'o3-mini' for complex questions"
    )

    reason: str = Field(
        description="Provide a detailed explanation of your reasoning behind categorising the question as either 'simple' or 'complex'."
    )


# ============================================================================
# AI MODELS INITIALIZATION
# ============================================================================

Complex_Chat_Model = ChatOpenAI(
    model='o3-mini',
    temperature=1.0,
    max_completion_tokens=100
)

Classification_Model = ChatOpenAI(
    model='gpt-4o',
    temperature=1.0,
    max_completion_tokens=300
)

Simple_Chat_model = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.1,
    max_completion_tokens=300
)

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

promptit = PromptTemplate(
    template='question: {query}',
    input_variables=['query']
)

# ============================================================================
# PARSERS AND CHAINS SETUP
# ============================================================================

# Structured output parser for classification
chatmodel_with_parser = Classification_Model.with_structured_output(QUESTIONSENTIMENT)

# String output parser
parser = StrOutputParser()


# Additional structured output model for answers
class modelname(BaseModel):
    reason: str = Field(description="Reasoning behind the choice")
    answer: str = Field(description="Answer to the quetsion")


complex_output_model = Complex_Chat_Model.with_structured_output(modelname)
simple_output_model = Simple_Chat_model.with_structured_output(modelname)

# Classification chain
classification_chain = RunnableSequence(promptit, chatmodel_with_parser)

# Processing chains for different complexity levels
complexChain = RunnableSequence(promptit, complex_output_model)
simplechain = RunnableSequence(promptit, simple_output_model)

# ============================================================================
# ROUTING LOGIC
# ============================================================================

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'simple'and x.model_names=='gpt-4o-mini', simplechain),
    (lambda x: x.sentiment == 'complex' and x.model_names=='o3-mini', complexChain),
    RunnableLambda(lambda x: "could not find sentiment")
)

# Final processing chain
final_chain = RunnableSequence(classification_chain, branch_chain)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test classification first
    test_query = 'How many days are there in a year?'
    classification_result = classification_chain.invoke({'query': test_query})

    print(f"Classification: {classification_result.sentiment}")
    print(f"Recommended Model: {classification_result.model_names}")
    print(f"Classification Reason: {classification_result.reason}")
    print("-" * 50)

    # Test full chain
    resultant = final_chain.invoke({'query': test_query})
    print(f"Final Answer: {resultant.answer}")
    print(f"Answer Reasoning: {resultant.reason}")