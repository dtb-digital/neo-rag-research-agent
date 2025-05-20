#!/usr/bin/env python3
"""
Minimal testskript for MCP-serveren.
Dette skriptet tester direkte kommunikasjon med MCP-serveren via stdin/stdout.
"""

import subprocess
import json
import sys
import time
import os
import re

# Finn riktig sti til mcp_server.py
current_dir = os.path.dirname(os.path.abspath(__file__))
mcp_server_path = os.path.join(current_dir, "mcp_server.py")

# Start MCP-serveren som subprocess
print(f"Starter MCP-server fra: {mcp_server_path}")
process = subprocess.Popen(
    [sys.executable, mcp_server_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Gi serveren tid til å starte
time.sleep(1)

if process.poll() is not None:
    print(f"MCP-server avsluttet med kode {process.returncode}")
    stderr = process.stderr.read()
    print(f"STDERR: {stderr}")
    sys.exit(1)

# Regex for å identifisere loggemeldinger
log_pattern = re.compile(r'^((\d{4}-\d{2}-\d{2})|(\[\d{2}/\d{2}/\d{2}))')

# Test semantic_search-verktøyet
try:
    print("Sender semantic_search-forespørsel...")
    request = {
        "name": "semantic_search",
        "params": {
            "query": "Hva er offentlighetsprinsippet?",
            "top_k": 3
        }
    }
    
    # Skriv til stdin
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    
    # Vent på respons med timeout
    print("Venter på respons...")
    start_time = time.time()
    timeout = 30  # 30 sekunder timeout
    
    # Les stderr parallelt for å se logger
    stderr_output = []
    got_response = False
    
    while time.time() - start_time < timeout and not got_response:
        # Sjekk om det er data tilgjengelig på stderr
        if process.stderr.readable():
            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_output.append(stderr_line.strip())
                print(f"STDERR: {stderr_line.strip()}")
        
        # Sjekk om det er data tilgjengelig på stdout
        if process.stdout.readable():
            stdout_line = process.stdout.readline().strip()
            if stdout_line:
                # Sjekk om det er en loggmelding
                if log_pattern.match(stdout_line):
                    print(f"LOGG: {stdout_line}")
                    continue
                
                # Hvis ikke en loggmelding, behandle som JSON-respons
                print(f"Rå respons: {stdout_line}")
                
                # Parse JSON-respons
                try:
                    response = json.loads(stdout_line)
                    print("Formatert respons:")
                    print(json.dumps(response, indent=2, ensure_ascii=False))
                    got_response = True
                except json.JSONDecodeError as e:
                    print(f"Kunne ikke parse JSON: {e}")
        
        # Kort pause før neste forsøk
        time.sleep(0.1)
    
    if not got_response:
        print("Timeout - ingen respons mottatt innen tidsfristen")
        if stderr_output:
            print("\nLogger fra stderr:")
            for line in stderr_output:
                print(line)
    
except Exception as e:
    print(f"Feil ved testing: {e}")
finally:
    print("Avslutter MCP-server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("MCP-server ville ikke avslutte, tvinger nedleggelse...")
        process.kill() 