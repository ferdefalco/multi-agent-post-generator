# multi-agent-post-generator
Multi-Agent Blog Post Generator with Ollama and DuckDuckGo

This project demonstrates a local multi-agent LLM system built with LangGraph, LangChain, and Ollama. It orchestrates multiple specialized agents to collaboratively generate multilingual blog content.

The workflow includes:

1) A Researcher agent that gathers up-to-date information using DuckDuckGo search tools

2) A Writer agent powered by a local LLaMA 3 model via Ollama to generate an original blog post in English

3) A Translator agent, also using LLaMA 3, to produce Portuguese and French versions of the content

