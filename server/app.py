import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import SimpleHandler, HTTPServer

def main():
    print("Starting OpenEnv Server...")
    server = HTTPServer(('0.0.0.0', 7860), SimpleHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
