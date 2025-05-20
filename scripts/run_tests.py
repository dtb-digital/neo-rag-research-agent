#!/usr/bin/env python3
"""
Kjør tester for Lovdata RAG-agent.

Dette skriptet kjører alle tilgjengelige tester for RAG-agenten for norske lover,
eller spesifikke tester etter ønske.

Bruk:
    python run_tests.py --all          # Kjør alle tester
    python run_tests.py --pinecone     # Kjør kun Pinecone-test
    python run_tests.py --mcp          # Kjør kun MCP-test
    python run_tests.py --suite        # Kjør test-suiten
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

# Tilgjengelige testskript
TEST_PINECONE = os.path.join(SCRIPT_DIR, "test_pinecone_simple.py")
TEST_MCP = os.path.join(SCRIPT_DIR, "test_mcp_simple.py")
TEST_SUITE = os.path.join(SCRIPT_DIR, "test_suite.py")

def run_command(command, timeout=120):
    """Kjør en kommando og returner resultatet."""
    print(f"\n{'='*80}")
    print(f"KJØRER: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        env = os.environ.copy()
        # Sett høyere loggnivå for testen
        env['LOG_LEVEL'] = 'DEBUG'
        
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
    """Kjør Pinecone-tilkoblingstest."""
    return run_command([sys.executable, TEST_PINECONE])

def run_mcp_test():
    """Kjør MCP-server-test."""
    return run_command([sys.executable, TEST_MCP], timeout=180)  # Øk timeout til 3 minutter

def run_test_suite(args=None):
    """Kjør test-suiten med valgfrie argumenter."""
    cmd = [sys.executable, TEST_SUITE]
    
    if args:
        cmd.extend(args)
    
    return run_command(cmd, timeout=180)  # Lengre timeout for test-suite

def main():
    """Hovedfunksjon."""
    parser = argparse.ArgumentParser(description="Kjør tester for Lovdata RAG-agent")
    
    # Legg til argumenter
    parser.add_argument("--all", action="store_true", help="Kjør alle tester")
    parser.add_argument("--pinecone", action="store_true", help="Kjør Pinecone-tilkoblingstest")
    parser.add_argument("--mcp", action="store_true", help="Kjør MCP-server-test")
    parser.add_argument("--suite", action="store_true", help="Kjør test-suiten")
    parser.add_argument("--suite-args", type=str, help="Argumenter til test-suiten (i anførselstegn)")
    
    args = parser.parse_args()
    
    # Hvis ingen tester er valgt, velg alle
    if not (args.all or args.pinecone or args.mcp or args.suite):
        args.all = True
    
    # Kjør valgte tester
    results = {}
    
    if args.all or args.pinecone:
        print("\nTester Pinecone-tilkobling...")
        results["pinecone"] = run_pinecone_test()
    
    if args.all or args.mcp:
        print("\nTester MCP-server...")
        results["mcp"] = run_mcp_test()
    
    if args.all or args.suite:
        print("\nKjører test-suite...")
        suite_args = []
        if args.suite_args:
            suite_args = args.suite_args.split()
        results["suite"] = run_test_suite(suite_args)
    
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