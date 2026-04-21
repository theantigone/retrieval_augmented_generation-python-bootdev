#!/usr/bin/env python3

from lib.semantic_search import verify_model,embed_text

import argparse

def main():
	parser = argparse.ArgumentParser(description="Semantic Search CLI")
	subparsers=parser.add_subparsers(dest='command',help='Available commands')

	subparsers.add_parser('verify',help='Verify the embedding model loads properly')

	embed_parser=subparsers.add_parser('embed_text',help='Embed the text')
	embed_parser.add_argument('text',type=str,help='Text to embed')

	args = parser.parse_args()

	match args.command:
		case 'embed_text':
			embed_text(args.text)
		case 'verify':
			verify_model()
		case _:
			parser.print_help()

if __name__ == "__main__":
	main()
