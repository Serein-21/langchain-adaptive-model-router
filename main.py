from os import system

from click import prompt
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableParallel,RunnableSequence,RunnableBranch
from typing import List,Annotated,Optional,Literal
from pydantic import BaseModel, Field
from sympy.polys.polyconfig import query

load_dotenv()

#Prompt Template
promptit=PromptTemplate(template='question: {query}',input_variables=['query'])



Complex_Chat_Model=ChatOpenAI(model='o3-mini',temperature=1.0,max_completion_tokens=100)

Classification_Model=ChatOpenAI(model='gpt-4o', temperature=1.0 ,max_completion_tokens=200)

Simple_Chat_model=ChatOpenAI(model='gpt-4.1-nano', temperature=0.1, max_completion_tokens=100)

class QUESTIONSENTIMENT(BaseModel):

    sentiment:Literal['simple','complex']=Field(default='simple',description= """"Classify the question complexity for optimal AI model routing:

SIMPLE: Questions requiring basic information retrieval, simple calculations, definitions, or straightforward explanations. Examples: "What is photosynthesis?", "Calculate 15% of 200", "Translate this to Spanish"

COMPLEX: Questions requiring multi-step reasoning, mathematical proofs, strategic analysis, algorithm design, scientific problem-solving, or deep domain expertise. Examples: "Prove mathematical theorems", "Design system architecture", "Analyze business strategy", "Solve multi-variable optimization problems"")""""")
    reason:str=Field(description="Provide a detailed explanation of your reasoning behind categorising the question as either 'simple' or 'complex'.")


chatmodel_with_parser=Classification_Model.with_structured_output(QUESTIONSENTIMENT)


finalprompt=promptit.format(query='How many days are there in a year?')

result=chatmodel_with_parser.invoke(finalprompt)

parser=StrOutputParser()

classification_chain=RunnableSequence(promptit,chatmodel_with_parser)


#print(result.sentiment)
#print(result.reason)

#Making two chains that will go inside Chain Branching
complexChain=RunnableSequence(promptit,Complex_Chat_Model,parser)
simplechain=RunnableSequence(promptit,Simple_Chat_model,parser)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'simple',simplechain),
    (lambda x:x.sentiment == 'complex',complexChain ),
    RunnableLambda(lambda x: "could not find sentiment")
)

final_chain=RunnableSequence(classification_chain,branch_chain)
resultant=final_chain.invoke(finalprompt)
print(resultant)


