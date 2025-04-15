# cli.py

import argparse
from generate import generate_text

def main():
     parser = argparse.ArgumentParser(description="🧠✌️ VibeAIr")
     parser.add_argument("prompt", type=str, help="Starting prompt for generation")
     parser.add_argument("--length", type=int, default=200, help="Number of characters to generate")

     args = parser.parse_args()
     result = generate_text(args.prompt, length=args.length)
     print("\nGenerated Text:\n")
     print(result)

if __name__ == "__main__":
     main()