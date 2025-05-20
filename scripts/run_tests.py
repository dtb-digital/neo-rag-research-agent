#!/usr/bin/env python3
"""
Kjør tester for Lovdata RAG-agent.

Dette skriptet kjører testene for RAG-agenten for norske lover.

Bruk:
    python run_tests.py --all     # Kjør alle tester
    python run_tests.py --pinecone # Kjør bare Pinecone-test
    python run_tests.py --mcp     # Kjør bare MCP-test
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Prosjektkonfigurasjon
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Testskripter
TEST_PINECONE = os.path.join(SCRIPT_DIR, "test_pinecone.py")
TEST_MCP = os.path.join(SCRIPT_DIR, "test_mcp.py")

def run_command(command, timeout=120):
    """Kjør en kommando og returner resultatet."""
    print(f"\n{'='*80}")
    print(f"KJØRER: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Sett opp miljøet for kommandoen
        env = os.environ.copy()
        
        # Kjør kommandoen
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
            env=env
        )
        
        duration = time.time() - start_time
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n{'='*80}")
            print(f"FEIL: Kommando avsluttet med kode {result.returncode}")
            print(f"Varighet: {duration:.2f} sekunder")
            print(f"{'='*80}\n")
            return False
        
        print(f"\n{'='*80}")
        print(f"SUKSESS: Kommando fullført")
        print(f"Varighet: {duration:.2f} sekunder")
        print(f"{'='*80}\n")
        
        return True
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"FEIL: Kommando timeout etter {timeout} sekunder")
        print(f"Varighet: {duration:.2f} sekunder")
        print(f"{'='*80}\n")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"FEIL: {str(e)}")
        print(f"Varighet: {duration:.2f} sekunder")
        print(f"{'='*80}\n")
        return False

def run_pinecone_test():
    """Kjør test av Pinecone-tilkobling."""
    return run_command([sys.executable, TEST_PINECONE])

def run_mcp_test(test_level="basic"):
    """Kjør test av MCP-serveren."""
    cmd = [sys.executable, TEST_MCP]
    
    if test_level != "basic":
        cmd.append(f"--level={test_level}")
        
    return run_command(cmd, timeout=180)  # 3 minutter timeout

def main():
    """Hovedfunksjon."""
    parser = argparse.ArgumentParser(description="Kjør tester for Lovdata RAG-agent")
    
    # Testutvalg
    parser.add_argument("--all", action="store_true", help="Kjør alle tester")
    parser.add_argument("--pinecone", action="store_true", help="Kjør Pinecone-tilkoblingstest")
    parser.add_argument("--mcp", action="store_true", help="Kjør MCP-servertest")
    
    # MCP testkonfigurasjon
    parser.add_argument("--mcp-level", choices=["minimal", "basic", "full"], default="basic",
                      help="Testnivå for MCP-testen (minimal, basic, full)")
    
    args = parser.parse_args()
    
    # Hvis ingen tester er valgt, kjør alle
    if not (args.all or args.pinecone or args.mcp):
        args.all = True
    
    # Lagre resultater
    results = {}
    
    # Kjør valgte tester
    if args.all or args.pinecone:
        print("Tester Pinecone-tilkobling...")
        results["pinecone"] = run_pinecone_test()
    
    if args.all or args.mcp:
        print("Tester MCP-server...")
        results["mcp"] = run_mcp_test(test_level=args.mcp_level)
    
    # Oppsummering
    print("\n" + "="*80)
    print("TESTRESULTATER")
    print("="*80)
    
    all_passed = True
    for test_name, success in results.items():
        status = "✅ BESTÅTT" if success else "❌ FEILET"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "-"*80)
    overall_status = "✅ ALLE TESTER BESTÅTT" if all_passed else "❌ NOEN TESTER FEILET"
    print(f"Samlet resultat: {overall_status}")
    print("="*80)
    
    # Returner exit-kode
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 