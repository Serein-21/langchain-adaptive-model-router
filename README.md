# Adaptive Model Router with Langchain

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Langchain](https://img.shields.io/badge/Langchain-Powered-brightgreen.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blueviolet.svg)](https://openai.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-yellow.svg)](https://pydantic-docs.helpmanual.io/)

## Problem Statement

When interacting with Large Language Models (LLMs), queries can vary significantly in complexity. Simpler queries can be handled efficiently by smaller, faster models, while complex queries require more powerful, and often more expensive, models to generate accurate and comprehensive responses. Using a single, powerful model for all types of queries can lead to unnecessary computational overhead and higher operational costs. Conversely, relying solely on a simpler model may result in suboptimal answers for complex tasks. This project addresses the need for an intelligent system that can dynamically route queries to the most appropriate LLM based on their inherent complexity, thereby optimizing for both performance and cost-efficiency.

## Overview

This project demonstrates an intelligent routing system for language model queries. It leverages Langchain to classify user questions as 'simple' or 'complex' and then routes them to the most appropriate OpenAI language model for an optimized response in terms of performance and cost.

The system uses a sophisticated classification model to determine the complexity of a query and then dynamically selects between a simpler, faster model for straightforward questions and a more powerful model for complex, multi-step reasoning tasks.

## Features

* **Intelligent Query Classification:** Automatically categorizes user questions into 'simple' or 'complex'.
* **Dynamic Model Routing:** Selects the optimal OpenAI model (e.g., `gpt-4.1-nano` for simple, `o3-mini` for complex) based on query classification.
* **Efficient Resource Utilization:** Optimizes for cost and speed by using simpler models for less demanding tasks.
* **Extensible Architecture:** Built with Langchain's `RunnableSequence` and `RunnableBranch` for easy modification and extension.
* **Structured Output:** Uses Pydantic for defining the classification schema, ensuring reliable output.

## How It Works

The core logic of the application is built around a series of Langchain Runnables:

1.  **Input Processing:** User queries are initially formatted using a `PromptTemplate`.
    ```python
    promptit = PromptTemplate(template='question: {query}', input_variables=['query']) #
    ```

2.  **Query Classification:**
    * A dedicated `Classification_Model` (e.g., `gpt-4o`) is used to analyze the query.
    * This model is combined with a Pydantic schema (`QUESTIONSENTIMENT`) to output a structured classification ('simple' or 'complex') along with the reasoning.
    ```python
    class QUESTIONSENTIMENT(BaseModel): #
        sentiment: Literal['simple', 'complex'] = Field(default='simple', description="...") #
        reason: str = Field(description="...") #

    chatmodel_with_parser = Classification_Model.with_structured_output(QUESTIONSENTIMENT) #
    classification_chain = RunnableSequence(promptit, chatmodel_with_parser) #
    ```
    * **Simple Questions:** Defined as those requiring basic information retrieval, simple calculations, definitions, or straightforward explanations.
        * *Examples:* "What is photosynthesis?", "Calculate 15% of 200", "Translate this to Spanish"
    * **Complex Questions:** Defined as those requiring multi-step reasoning, mathematical proofs, strategic analysis, algorithm design, scientific problem-solving, or deep domain expertise.
        * *Examples:* "Prove mathematical theorems", "Design system architecture", "Analyze business strategy", "Solve multi-variable optimization problems"

3.  **Conditional Model Routing (Branching):**
    * Based on the `sentiment` from the classification step, a `RunnableBranch` directs the query to one of two chains:
        * `simplechain`: Uses a `Simple_Chat_model` (e.g., `gpt-4.1-nano`) for simple queries.
        * `complexChain`: Uses a `Complex_Chat_Model` (e.g., `o3-mini`) for complex queries.
    ```python
    complexChain = RunnableSequence(promptit, Complex_Chat_Model, StrOutputParser()) #
    simplechain = RunnableSequence(promptit, Simple_Chat_model, StrOutputParser()) #

    branch_chain = RunnableBranch( #
        (lambda x: x.sentiment == 'simple', simplechain), #
        (lambda x: x.sentiment == 'complex', complexChain), #
        RunnableLambda(lambda x: "could not find sentiment") #
    )
    ```

4.  **Final Execution:** The entire process is encapsulated in a `final_chain` that first runs the classification and then the appropriate branched model chain.
    ```python
    final_chain = RunnableSequence(classification_chain, branch_chain) #
    resultant = final_chain.invoke(finalprompt) # finalprompt is the formatted user query #
    ```

## Technologies Used

* **Python:** Core programming language.
* **Langchain:** Framework for developing applications powered by language models.
    * [Langchain Documentation](https://python.langchain.com/)
    * `RunnableSequence`: For chaining components. [Docs](https://python.langchain.com/docs/expression_language/interface#runnablesequence)
    * `RunnableBranch`: For conditional logic. [Docs](https://python.langchain.com/docs/expression_language/how_to/routing#routerunnablebranch)
    * `PromptTemplate`: For managing prompts. [Docs](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
    * `ChatOpenAI`: For interacting with OpenAI models. [Docs](https://python.langchain.com/docs/integrations/chat/openai/)
* **OpenAI API:** Provides access to powerful language models like GPT-4o, o3-mini, and gpt-4.1-nano.
    * [OpenAI API Documentation](https://beta.openai.com/docs/)
* **Pydantic:** Data validation and settings management using Python type annotations.
    * [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
* **python-dotenv:** For managing environment variables (e.g., API keys).
    * [python-dotenv on PyPI](https://pypi.org/project/python-dotenv/)

## Setup and Usage

### Prerequisites

* Python 3.7+
* An OpenAI API key

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create a `requirements.txt` file listing `langchain-core`, `langchain-openai`, `python-dotenv`, `pydantic`, `click`, `sympy` - though sympy seems imported but not used directly in the core logic shown, and `openai` if `langchain-openai` doesn't pull it transitively.)

    **Example `requirements.txt`:**
    ```
    langchain-core
    langchain-openai
    python-dotenv
    pydantic
    click
    openai
    # sympy # if actually used
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root of your project directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

### Running the Script

Execute the `main.py` script:
```bash
python main.py
