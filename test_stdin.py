#!/usr/bin/env python
import json
import sys
import time

# Lag en spørring
query = 'Hva sier offentlighetsloven om innsyn i dokumenter?'
request = {
    'type': 'function_call',
    'function': {
        'name': 'sok_i_lovdata',
        'arguments': json.dumps({
            'sporsmal': query,
            'antall_resultater': 3
        })
    }
}

# Konverter til JSON og skriv til stdout
print(json.dumps(request))
sys.stdout.flush()

# Vent på svar
print("Venter på svar...", file=sys.stderr)
for line in sys.stdin:
    try:
        response = json.loads(line)
        print("Fikk svar:", file=sys.stderr)
        print(json.dumps(response, indent=2), file=sys.stderr)
        break
    except json.JSONDecodeError:
        print(f"Ugyldig JSON-svar: {line}", file=sys.stderr)

print("Ferdig.", file=sys.stderr) 