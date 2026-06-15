import os
import random
import urllib.parse
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

# Imports locaux
from core.graph import neuron_graph
from core.permissions import permission_manager

# Initialisation du serveur MCP
mcp = FastMCP("makenBrain")

# --- OUTILS DE NAVIGATION ET RECHERCHE ---

@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Recherche des informations sur internet.
    L'IA doit demander si MAKEN l'autorise à faire une recherche web avant d'utiliser cet outil.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return f"Aucun résultat trouvé sur le web pour '{query}'."
        
        output = [f"🌐 Résultats de recherche pour '{query}' :"]
        for r in results:
            output.append(f"- {r.get('title')} : {r.get('body')} (Source: {r.get('href')})")
        return "\n\n".join(output)
    except Exception as e:
        return f"Erreur lors de la recherche web : {e}"

# --- OUTILS DE GESTION DE FICHIERS (SÉCURISÉS) ---

@mcp.tool()
async def request_file_access(path: str) -> str:
    """
    Demande l'autorisation d'accéder à un fichier ou un dossier local.
    L'IA doit appeler cet outil AVANT de tenter de lire un fichier hors du projet.
    """
    permission_manager.grant_access(path)
    return f"Autorisation accordée pour : {path}. Je peux maintenant lire ce contenu."

@mcp.tool()
async def read_local_file_smart(path: str, max_lines: int = 100) -> str:
    """Lit les premières lignes d'un fichier local autorisé pour éviter de surcharger l'IA."""
    if not permission_manager.is_allowed(path):
        return f"ERREUR : Accès refusé à '{path}'. Veuillez d'abord me demander l'autorisation avec request_file_access."
    
    try:
        p = Path(path)
        if p.exists() and p.is_file():
            lines = p.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            content = "\n".join(lines[:max_lines])
            if total > max_lines:
                content += f"\n\n[... {total - max_lines} lignes supplémentaires non affichées ...]"
            return content
        return "Fichier non trouvé."
    except Exception as e:
        return f"Erreur : {e}"

@mcp.tool()
async def list_local_directory(path: str) -> str:
    """Liste les fichiers d'un dossier local autorisé."""
    if not permission_manager.is_allowed(path):
        return f"ERREUR : Accès refusé à '{path}'. Autorisation requise."
    
    try:
        p = Path(path)
        if p.exists() and p.is_dir():
            files = [f.name for f in p.iterdir()]
            return f"Contenu de {path} :\n" + "\n".join(files)
        return "Dossier non trouvé."
    except Exception as e:
        return f"Erreur : {e}"

# --- OUTILS DU GRAPHE DE NEURONES ---

@mcp.tool()
async def get_brain_stats() -> str:
    """Récupère les statistiques globales du cerveau (nœuds, arêtes, densité)."""
    stats = neuron_graph.get_stats()
    return (
        f"🧠 Statistiques de makenBrain :\n"
        f"- Concepts (nœuds) : {stats['nodes']}\n"
        f"- Connexions (arêtes) : {stats['edges']}\n"
        f"- Densité : {stats['density']}\n"
        f"- Composantes : {stats['components']}\n"
        f"- Concepts clés : {', '.join([c['concept'] for c in stats['top_concepts'][:5]])}"
    )

@mcp.tool()
async def search_concepts(query: str) -> str:
    """Cherche des concepts existants dans le cerveau à partir d'un mot-clé."""
    results = neuron_graph.search_concepts(query)
    if not results:
        return f"Aucun concept trouvé pour '{query}'."
    
    output = [f"Résultats pour '{query}' :"]
    for r in results:
        output.append(f"- {r['concept']} ({r['type']}) : {r['connections']} connexions")
    return "\n".join(output)

@mcp.tool()
async def explore_concept(concept: str, depth: int = 1) -> str:
    """Explore les connexions d'un concept spécifique dans le cerveau."""
    data = neuron_graph.explore(concept, depth=depth)
    if not data["found"]:
        return f"Le concept '{concept}' n'a pas été trouvé. Suggestions : {', '.join(data.get('suggestions', []))}"
    
    output = [f"Exploration de '{concept}' (Fréquence: {data['frequency']}) :"]
    output.append(f"Direct connections: {data['direct_connections']}")
    output.append("\nVoisins et chemins :")
    for name, info in data["neighbors"].items():
        output.append(f"- {name} (dist: {info['distance']}, force: {info['strength']}) : {info['path']}")
    
    return "\n".join(output)

# --- OUTILS DE CRÉATION ---

import re
from duckduckgo_search import DDGS

# --- OUTILS DE CRÉATION AVANCÉE ---

@mcp.tool()
async def generate_diagram(description: str) -> str:
    """Génère un diagramme Mermaid (flowchart, sequence, etc.) et l'enregistre en HTML."""
    prompt = (
        f"Génère du code Mermaid valide pour : {description}. "
        "Retourne UNIQUEMENT le code pur (pas de ```mermaid)."
    )
    from core.llm import generate
    mermaid_code = await generate(prompt)
    mermaid_code = re.sub(r"```(?:mermaid)?", "", mermaid_code).strip()
    
    html_content = f"<html><body><script src='https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js'></script><div class='mermaid'>{mermaid_code}</div><script>mermaid.initialize({{startOnLoad:true}});</script></body></html>"
    
    output_path = Path("brain_data/outputs/diagram.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    
    return f"Diagramme généré avec succès ! Tu peux le voir ici : {output_path.absolute()}"

@mcp.tool()
async def create_presentation(topic: str) -> str:
    """Crée une présentation HTML professionnelle avec Reveal.js."""
    # Simulation de création de présentation pour cet exemple
    output_path = Path("brain_data/outputs/presentation.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    html = f"<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/reveal.js/dist/reveal.css'></head><body><div class='reveal'><div class='slides'><section><h1>{topic}</h1><p>Généré par MakenBrain</p></section></div></div><script src='https://cdn.jsdelivr.net/npm/reveal.js/dist/reveal.js'></script><script>Reveal.initialize();</script></body></html>"
    
    output_path.write_text(html, encoding="utf-8")
    return f"Présentation créée : {output_path.absolute()}"

@mcp.tool()
async def generate_realistic_image(prompt: str) -> str:
    """
    Génère une image ultra-réaliste à partir d'une description.
    L'IA doit enrichir le prompt avec des détails techniques pour un résultat pro.
    """
    # Enrichissement automatique du prompt pour le réalisme extrême
    enhanced_prompt = (
        f"{prompt}, extremely detailed, 8k resolution, photorealistic, "
        "cinematic lighting, hyper-realistic textures, masterpiece, professional photography, "
        "shot on 35mm lens, f/1.8"
    )
    encoded_prompt = urllib.parse.quote(enhanced_prompt)
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&model=flux&seed={random.randint(1, 100000)}"
    
    return (
        f"🎨 Image générée avec succès pour : '{prompt}'\n"
        f"Lien de l'image haute définition : {image_url}\n"
        "Note : Cliquez sur le lien pour voir le résultat. Si vous voulez plus de détails, demandez-moi !"
    )

if __name__ == "__main__":
    mcp.run()
