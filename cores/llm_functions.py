from pydantic import BaseModel
from typing import Optional, Type

from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from settings import settings


def call_llm(prompt_template : str,
             prompt_inputs   : dict,
             template_type   : Optional[str] = "jinja2",
             **kwargs) -> str:
    """
    Call Azure OpenAI GPT-4.1-mini model using LangChain

    Args:
        prompt_template: The prompt template string
        prompt_inputs: Dictionary of input variables for the template
        template_type: Type of template ("jinja2" or "f-string")

    Returns:
        str: The generated response from the model
    """

    # Initialize Azure OpenAI client
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_deployment_name,
        openai_api_version=settings.azure_openai_api_version,
        openai_api_key=settings.azure_openai_api_key,
        **kwargs
    )

    # Create the prompt using the template
    if template_type == "jinja2":
        prompt = PromptTemplate.from_template(prompt_template, template_format="jinja2")
        formatted_prompt = prompt.invoke(prompt_inputs)
    else:
        prompt = PromptTemplate(template=prompt_template, input_variables=list(prompt_inputs.keys()))
        formatted_prompt = prompt.invoke(prompt_inputs)

    # Call the model
    response = llm.invoke(formatted_prompt)

    return response.content


def call_structured_llm(prompt_template: str,
                        prompt_inputs: dict,
                        template_type: Optional[str] = "jinja2",
                        output_schema: Optional[Type[BaseModel]] = None,
                        **kwargs) -> dict:
    """
    Call Azure OpenAI GPT-4.1-mini model with structured output using LangChain

    Args:
        prompt_template: The prompt template string
        prompt_inputs: Dictionary of input variables for the template
        template_type: Type of template ("jinja2" or "f-string")
        output_schema: Pydantic model class for structured output

    Returns:
        dict: The structured response parsed according to the output schema
    """

    # Initialize Azure OpenAI client
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_deployment_name,
        openai_api_version=settings.azure_openai_api_version,
        openai_api_key=settings.azure_openai_api_key,
        **kwargs
    )

    # Create the prompt using the template
    if template_type == "jinja2":
        prompt = PromptTemplate.from_template(prompt_template, template_format="jinja2")
        formatted_prompt = prompt.invoke(prompt_inputs)
    else:
        prompt = PromptTemplate(template=prompt_template, input_variables=list(prompt_inputs.keys()))
        formatted_prompt = prompt.invoke(prompt_inputs)

    llm = llm.with_structured_output(output_schema)

    response = llm.invoke(formatted_prompt).model_dump()

    return response


if __name__ == "__main__":


    class Person(BaseModel):
        name: str
        email: str


    print(call_structured_llm(
        prompt_template="extract the name and email from the following text: {{ text }}",
        prompt_inputs={"text": "my name is John Doe and my email is john.doe@example.com"},
        template_type="jinja2",
        output_schema=Person,
    ))