#!/usr/bin/env python3
"""
Teste isolado do ActionExecutor
"""
import sys
import os

# Adicionar diretório src ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from mia.tools.action_executor import ActionExecutor

def main():
    print("🤖 Teste Isolado do M.I.A ActionExecutor")
    print("=" * 50)
    
    # Criar instância
    executor = ActionExecutor()
    print("✅ ActionExecutor inicializado")
    
    # Teste 1: Make note
    print("\n📝 Teste 1: Criando nota...")
    result1 = executor.make_note("Teste isolado do agente M.I.A", "Agent Test")
    print(f"Resultado: {result1}")
    
    # Teste 2: Create file
    print("\n📄 Teste 2: Criando arquivo...")
    result2 = executor.create_file("agent_isolated_test.txt", "Conteúdo criado pelo agente M.I.A\nTeste isolado funcionando!")
    print(f"Resultado: {result2}")
    
    # Verificar se arquivo foi criado
    if os.path.exists("agent_isolated_test.txt"):
        print("✅ Arquivo criado com sucesso!")
        with open("agent_isolated_test.txt", "r") as f:
            content = f.read()
            print(f"Conteúdo:\n{content}")
    else:
        print("❌ Arquivo não foi criado")
    
    print("\n🎯 Teste concluído!")

if __name__ == "__main__":
    main()
