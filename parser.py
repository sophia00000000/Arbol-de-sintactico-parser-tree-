import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

class EarleyItem:
    def __init__(self, rule, dot_pos, start_pos, end_pos=None):
        self.rule = rule  # (left_side, right_side)
        self.dot_pos = dot_pos
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.children = []  # Para construir el árbol
        
    def __str__(self):
        left, right = self.rule
        right_str = ' '.join(right[:self.dot_pos] + ['•'] + right[self.dot_pos:])
        return f"{left} → {right_str} [{self.start_pos}, {self.end_pos}]"
    
    def __eq__(self, other):
        return (self.rule == other.rule and 
                self.dot_pos == other.dot_pos and 
                self.start_pos == other.start_pos)
    
    def __hash__(self):
        return hash((self.rule, self.dot_pos, self.start_pos))
    
    def is_complete(self):
        return self.dot_pos >= len(self.rule[1])
    
    def next_symbol(self):
        if self.is_complete():
            return None
        return self.rule[1][self.dot_pos]

class EarleyParser:
    def __init__(self, grammar_file):
        self.grammar = self.load_grammar(grammar_file)
        self.start_symbol = None
        self.terminals = set()
        self.non_terminals = set()
        self.analyze_grammar()
        
    def load_grammar(self, filename):
        grammar = defaultdict(list)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parsear regla: A → B C D
                    if '→' in line:
                        left, right = line.split('→', 1)
                        left = left.strip()
                        right = right.strip().split()
                        grammar[left].append(right)
                    elif '->' in line:  # Permitir también ->
                        left, right = line.split('->', 1)
                        left = left.strip()
                        right = right.strip().split()
                        grammar[left].append(right)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {filename}")
            return {}
        
        return dict(grammar)
    
    def analyze_grammar(self):
        if not self.grammar:
            return
            
        # El símbolo de inicio es el primer no terminal
        self.start_symbol = next(iter(self.grammar.keys()))
        
        # Identificar no terminales y terminales
        self.non_terminals = set(self.grammar.keys())
        
        for productions in self.grammar.values():
            for production in productions:
                for symbol in production:
                    if symbol not in self.non_terminals:
                        self.terminals.add(symbol)
    
    def tokenize(self, input_string):
        # Tokenizador simple que maneja operadores y números
        tokens = []
        i = 0
        while i < len(input_string):
            char = input_string[i]
            if char.isspace():
                i += 1
                continue
            elif char.isdigit():
                # Leer número completo
                num = ''
                while i < len(input_string) and input_string[i].isdigit():
                    num += input_string[i]
                    i += 1
                tokens.append(('num', num))
            elif char in '+-*/()':
                # Mapear operadores a tokens de la gramática
                if char == '+':
                    tokens.append(('op_suma', char))
                elif char == '*':
                    tokens.append(('op_mul', char))
                elif char == '(':
                    tokens.append(('pari', char))
                elif char == ')':
                    tokens.append(('pard', char))
                elif char == '-':
                    tokens.append(('op_suma', char))  # Tratar - como op_suma
                elif char == '/':
                    tokens.append(('op_mul', char))   # Tratar / como op_mul
                i += 1
            elif char.isalpha():
                # Leer identificador
                ident = ''
                while i < len(input_string) and (input_string[i].isalnum() or input_string[i] == '_'):
                    ident += input_string[i]
                    i += 1
                tokens.append(('id', ident))
            else:
                i += 1
        
        return tokens
    
    def parse(self, input_string):
        tokens = self.tokenize(input_string)
        token_types = [token[0] for token in tokens]
        
        if not self.start_symbol:
            return False, None
        
        # Inicializar chart
        chart = [[] for _ in range(len(token_types) + 1)]
        
        # Predicción inicial
        for production in self.grammar.get(self.start_symbol, []):
            item = EarleyItem((self.start_symbol, production), 0, 0)
            chart[0].append(item)
        
        # Procesar cada posición
        for i in range(len(token_types) + 1):
            j = 0
            while j < len(chart[i]):
                item = chart[i][j]
                
                if item.is_complete():
                    # Completion
                    self.complete(chart, item, i)
                else:
                    next_sym = item.next_symbol()
                    if next_sym in self.non_terminals:
                        # Prediction
                        self.predict(chart, next_sym, i)
                    elif i < len(token_types) and next_sym == token_types[i]:
                        # Scan
                        self.scan(chart, item, i, tokens[i])
                
                j += 1
        
        # Verificar si hay una derivación completa
        for item in chart[len(token_types)]:
            if (item.rule[0] == self.start_symbol and 
                item.is_complete() and 
                item.start_pos == 0):
                return True, self.build_tree(chart, len(token_types), tokens)
        
        return False, None
    
    def predict(self, chart, non_terminal, pos):
        for production in self.grammar.get(non_terminal, []):
            new_item = EarleyItem((non_terminal, production), 0, pos)
            if new_item not in chart[pos]:
                chart[pos].append(new_item)
    
    def scan(self, chart, item, pos, token):
        new_item = EarleyItem(item.rule, item.dot_pos + 1, item.start_pos, pos + 1)
        new_item.children = item.children + [token]
        chart[pos + 1].append(new_item)
    
    def complete(self, chart, completed_item, pos):
        for item in chart[completed_item.start_pos]:
            if (not item.is_complete() and 
                item.next_symbol() == completed_item.rule[0]):
                new_item = EarleyItem(item.rule, item.dot_pos + 1, item.start_pos, pos)
                new_item.children = item.children + [completed_item]
                if new_item not in chart[pos]:
                    chart[pos].append(new_item)
    
    def build_tree(self, chart, final_pos, tokens):
        # Encontrar el item de inicio completo
        start_item = None
        for item in chart[final_pos]:
            if (item.rule[0] == self.start_symbol and 
                item.is_complete() and 
                item.start_pos == 0):
                start_item = item
                break
        
        if not start_item:
            return None
        
        # Construir el grafo
        G = nx.DiGraph()
        node_id = [0]  # Para generar IDs únicos
        
        def add_node_to_graph(item, parent_id=None):
            current_id = node_id[0]
            node_id[0] += 1
            
            if isinstance(item, tuple):  # Es un token terminal
                token_type, token_value = item
                G.add_node(current_id, label=token_value, type='terminal')
            else:  # Es un item no terminal
                G.add_node(current_id, label=item.rule[0], type='non_terminal')
                
                # Agregar hijos
                for child in item.children:
                    child_id = add_node_to_graph(child, current_id)
                    G.add_edge(current_id, child_id)
            
            return current_id
        
        root_id = add_node_to_graph(start_item)
        return G
    
    def visualize_tree(self, tree):
        if not tree:
            print("No hay árbol para visualizar")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Usar layout jerárquico
        pos = self._hierarchical_layout(tree)
        
        # Dibujar nodos
        node_colors = []
        node_labels = {}
        
        for node_id in tree.nodes():
            node_data = tree.nodes[node_id]
            node_labels[node_id] = node_data['label']
            
            if node_data['type'] == 'terminal':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightcoral')
        
        nx.draw(tree, pos, 
                with_labels=True, 
                labels=node_labels,
                node_color=node_colors,
                node_size=1500,
                font_size=10,
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                arrowsize=20)
        
        plt.title("Árbol de Sintaxis")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _hierarchical_layout(self, G):
        """Crear un layout jerárquico para el árbol"""
        pos = {}
        levels = {}
        
        # Encontrar el nivel de cada nodo
        def assign_levels(node, level):
            levels[node] = level
            for child in G.successors(node):
                assign_levels(child, level + 1)
        
        # Encontrar la raíz (nodo sin predecesores)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if roots:
            assign_levels(roots[0], 0)
        
        # Agrupar nodos por nivel
        level_groups = defaultdict(list)
        for node, level in levels.items():
            level_groups[level].append(node)
        
        # Asignar posiciones
        for level, nodes in level_groups.items():
            for i, node in enumerate(nodes):
                x = i - len(nodes) / 2 + 0.5
                y = -level
                pos[node] = (x, y)
        
        return pos

def main():
    # Crear el archivo de gramática de ejemplo si no existe
    grammar_content = """E → E op_suma T
E → T
T → T op_mul F
T → F
F → id
F → num
F → pari E pard"""
    
    try:
        with open('gra.txt', 'r') as f:
            pass
    except FileNotFoundError:
        print("Creando archivo gra.txt con gramática de ejemplo...")
        with open('gra.txt', 'w', encoding='utf-8') as f:
            f.write(grammar_content)
    
    # Crear parser
    parser = EarleyParser('gra.txt')
    
    if not parser.grammar:
        print("Error: No se pudo cargar la gramática")
        return
    
    print("Gramática cargada:")
    for left, productions in parser.grammar.items():
        for production in productions:
            print(f"{left} → {' '.join(production)}")
    
    print(f"\nSímbolo de inicio: {parser.start_symbol}")
    print(f"No terminales: {parser.non_terminals}")
    print(f"Terminales: {parser.terminals}")
    
    # Probar con entrada del usuario
    while True:
        try:
            input_string = input("\nIngrese una expresión (o 'quit' para salir): ").strip()
            if input_string.lower() == 'quit':
                break
            
            print(f"\nAnalizando: {input_string}")
            
            # Tokenizar y mostrar tokens
            tokens = parser.tokenize(input_string)
            print(f"Tokens: {tokens}")
            
            # Parsear
            success, tree = parser.parse(input_string)
            
            if success:
                print("ACEPTA")
                print("Árbol de sintaxis generado exitosamente")
                
                # Visualizar árbol
                parser.visualize_tree(tree)
                
                # Mostrar información del árbol
                if tree:
                    print(f"Nodos en el árbol: {tree.number_of_nodes()}")
                    print(f"Aristas en el árbol: {tree.number_of_edges()}")
            else:
                print("NO ACEPTA")
                print("La cadena no es válida según la gramática")
                
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()