# Multi-Agent Blog Post Generator using Ollama:
# 1. Research latest news about a topic (DuckDuckGo)
# 2. Write a short blog post in English using Ollama
# 3. Translate it into Portuguese and French using Ollama
# Flow: Start -> Researcher -> Writer -> Translator -> End

# run in terminal: ollama serve 
# run in terminal: ollama pull llama3

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class AgentState(TypedDict):
    topic: str
    research_data: List[str]  
    blog_post: str            

def researcher_node(state: AgentState):
    topic = state["topic"]
    print(f"Researcher is looking for latest news about: {topic}")
    
    search = DuckDuckGoSearchRun()
    
    try:
        results = search.run(f"latest news and facts about {topic}")
    except Exception as e:
        results = f"Could not find data: {e}"
        
    print("Research complete.")
    
    return {"research_data": state.get("research_data", []) + [results]}

def writer_node(state: AgentState):
    print("Writer is writing the draft...")
    
    topic = state["topic"]
    data = state["research_data"][-1] if state["research_data"] else ""
    
    llm = ChatOllama(model="llama3", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a blog writer. 
           Write a short, engaging blog post about "{topic}" 
           based only on the following research data:
           {data} Return just the blog post content."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"topic": topic, "data": data})
    
    print("Writing complete.")
    return {"blog_post": response.content}

def translator_node(state: AgentState):
    print("Translator is translating to Portuguese and French...")
    
    blog_post = state["blog_post"] if state["blog_post"] else ""
    
    llm = ChatOllama(model="llama3", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a translator. 
           Translate the following text to Portuguese and French: "{blog_post}" 
           Return the blog post content in the 3 languages, the original in English plus the Portuguese and French version."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"blog_post": blog_post})
    
    print("Translating complete.")
    return {"blog_post": response.content}

# Building the LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Translator", translator_node)

# Flow: Start -> Researcher -> Writer -> Translator -> END
workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Writer")
workflow.add_edge("Writer","Translator")
workflow.add_edge("Translator", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Starting the Multi-Agent System...\n")
    
    inputs: AgentState = {
        "topic": "Electricity Markets in Alberta",
        "research_data": [],
        "blog_post": "",
    }
    
    result = app.invoke(inputs)
    
    print("\n---------------- FINAL OUTPUT ----------------\n")
    print(result["blog_post"])

# Output Example
# Starting the Multi-Agent System...

# Researcher is looking for latest news about: Electricity Markets in Alberta
# Research complete.
# Writer is writing the draft...
# Writing complete.
# Translator is translating to Portuguese and French...
# Translating complete.

# ---------------- FINAL OUTPUT ----------------

# Here are the translations:

# **English (Original)**

# **Rebooting Alberta's Electricity Markets: A Wake-Up Call for a Changing Landscape**

# As we face a future dominated by rising demand and renewables, it's time to rethink the way we approach electricity markets in Alberta. The latest report from the International Energy Agency (IEA) serves as a stark reminder that our current systems are woefully inadequate.

# In Alberta specifically, the need for an overhaul is clear. With the increasing reliance on solar power and other renewable energy sources, our traditional grid infrastructure is struggling to keep up. Moreover, the rising demand for electricity due to population growth and economic development only exacerbates the issue.

# But what's the solution? For one, manual shopping success depends heavily on market timing. In Alberta, electricity rates typically reach their lowest points in late winter and early spring – a window of opportunity that savvy consumers can capitalize on by renewing their contracts accordingly.

# However, this approach is far from foolproof. It requires an intimate understanding of the complex dynamics at play in the market, as well as the ability to time the renewal process perfectly. For many consumers, this level of expertise is simply out of reach.

# That's why it's crucial that we adopt a more forward-thinking approach to electricity markets in Alberta. One that prioritizes innovation and flexibility, rather than relying on outdated systems that are ill-equipped to handle the demands of tomorrow.

# The IEA report provides valuable insights into the global trends shaping the future of energy. For instance, did you know that Europe's electricity market is characterized by a diverse array of energy sources? The main source of electricity across the continent varies by region, reflecting different geographical and economic factors.

# As we look to the future, it's clear that Alberta must follow suit. By embracing a more decentralized and adaptable approach to electricity markets, we can ensure a sustainable and reliable supply of power for generations to come.

# The time to act is now. It's high time for Alberta to reboot its electricity markets and join the 21st century.

# **Portuguese**

# **Reiniciando os Mercados de Energia Elétrica da Alberta: Um Chamado à Atenção para um Novo Landscape**

# Ao enfrentar o futuro dominado por uma demanda crescente e fontes renováveis, é hora de repensar como abordamos os mercados de energia elétrica na Alberta. O mais recente relatório da Agência Internacional de Energia (AIE) serve como um lembrete severo de que nossos sistemas atuais são lamentavelmente inadequados.

# Na Alberta em específico, a necessidade de uma revisão é clara. Com o aumento da dependência de energia solar e outras fontes renováveis, nossa infraestrutura tradicional de rede elétrica está lutando para manter o ritmo. Além disso, a demanda crescente por energia elétrica devido ao crescimento populacional e ao desenvolvimento econômico apenas agrava o problema.

# Mas qual é a solução? Para começar, o sucesso no shopping manual depende fortemente do timing do mercado. Na Alberta, as taxas de energia elétrica tendem a alcançar seus pontos mais baixos no final do inverno e no início da primavera – uma janela de oportunidade que consumidores sábios podem capitalizar ao renovarem seus contratos adequadamente.

# No entanto, essa abordagem é longe de ser infalível. Ela requer um conhecimento íntimo das dinâmicas complexas em jogo no mercado, bem como a habilidade para timing o processo de renovação perfeitamente. Para muitos consumidores, esse nível de expertise está simplesmente fora do alcance.

# É por isso que é crucial que adotemos uma abordagem mais pensativa e flexível para os mercados de energia elétrica na Alberta. Uma que priorize a inovação e a flexibilidade, em vez de se apoiar em sistemas obsoletos que não estão preparados para lidar com as demandas do futuro.

# O relatório da AIE fornece valiosas informações sobre os tendências globais que moldam o futuro da energia. Por exemplo, você sabia que o mercado de energia elétrica da Europa é caracterizado por uma variedade diversa de fontes energéticas? A principal fonte de energia elétrica em toda a região varia dependendo do local, refletindo diferentes fatores geográficos e econômicos.

# Ao olhar para o futuro, é claro que a Alberta deve seguir esse caminho. Ao abraçar uma abordagem mais descentralizada e adaptável para os mercados de energia elétrica, podemos assegurar uma fornecedora sustentável e confiável de energia elétrica para gerações futuras.

# É hora de agir agora. É alta hora para a Alberta reiniciar seus mercados de energia elétrica e se juntar ao século 21.

# **French**

# **Réinitialiser les marchés d'énergie électrique de l'Alberta : Un appel à réflexion pour un paysage en évolution**

# En face d'un avenir dominé par une demande croissante et des sources renouvelables, il est temps de repenser la façon dont nous abordons les marchés d'énergie électrique dans l'Alberta. Le plus récent rapport de l'Agence internationale de l'énergie (AIE) nous rappelle sévèrement que nos systèmes actuels sont lamentablement inadequats.

# En particulier en Alberta, le besoin d'une révision est clair. Avec une dépendance croissante à l'énergie solaire et à d'autres sources renouvelables, notre infrastructure traditionnelle de réseau électrique a du mal à suivre le rythme. De plus, la demande croissante pour l'énergie électrique due au croissance démographique et au développement économique ne fait qu'aggraver le problème.

# Mais qu'est-ce que la solution ? Pour commencer, le succès dans le shopping manuel dépend fortement du timing du marché. En Alberta, les tarifs de l'énergie électrique tendent à atteindre leurs points bas en fin d'hiver et au début du printemps - une fenêtre d'opportunité que les consommateurs avisés peuvent capitaliser en renouvelant leurs contrats de manière appropriée.

# Cependant, cette approche n'est pas loin d'être infaillible. Elle nécessite un understanding intime des dynamiques complexes en jeu sur le marché, ainsi qu'une capacité à timing le processus de renouvellement parfaitement. Pour de nombreux consommateurs, ce niveau d'expertise est simplement hors de portée.

# C'est pourquoi il est crucial que nous adoptions une approche plus pensante et flexible pour les marchés d'énergie électrique en Alberta. Une qui priorise l'innovation et la flexibilité, au lieu de se appuyer sur des systèmes obsolètes qui ne sont pas préparés à gérer les demandes du futur.

# Le rapport de l'AIE fournit des informations précieuses sur les tendances mondiales qui forgent le futur de l'énergie. Par exemple, savez-vous que le marché de l'énergie électrique en Europe est caractérisé par une variété diverse de sources énergétiques ? La principale source d'énergie électrique dans tout le continent varie en fonction du lieu, reflétant différents facteurs géographiques et économiques.

# En regardant vers l'avenir, il est clair que l'Alberta doit suivre ce chemin. En embrassant une approche plus décentralisée et adaptable pour les marchés d'énergie électrique, nous pouvons assurer un approvisionnement durable et fiable en énergie électrique pour les générations à venir.

# Il est temps d'agir maintenant. Il est haute fois que l'Alberta réinitialise ses marchés d'énergie électrique et se joint au XXIe siècle.
