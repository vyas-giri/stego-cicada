import argparse
from steganography import hide_message, extract_message

def main():
    parser = argparse.ArgumentParser(description="Steganography Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hide_parser = subparsers.add_parser("hide", help="Hide a message in an image")
    hide_parser.add_argument("image_path", type=str, help="Path to the input image")
    hide_parser.add_argument("message", type=str, help="Message to hide in the image")
    hide_parser.add_argument("--password", type=str,  help="Password for encryption")
    
    extract_parser = subparsers.add_parser("extract", help="Extract a message from an image")
    extract_parser.add_argument("image_path", type=str, help="Path to the image with hidden message")
    extract_parser.add_argument("-p", "--password", type=str, help="Password used for encryption")
    args = parser.parse_args()

    if args.command == "hide":
        res: dict = hide_message(args.image_path, args.message, args.password)
        print(f"Message hidden successfully.\nResult: {res}")
    elif args.command == "extract":
        message = extract_message(args.image_path, args.password)
        print(f"Extracted message:\n{message}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()