import argparse
import sys

from stego.steganography import hide_message, extract_message
from stego.steganalysis import analyze_jpeg_dct, analyze_png_lsb, analyze_image

def main():
    parser = argparse.ArgumentParser(description="Steganography Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hide_parser = subparsers.add_parser("hide", help="Hide a message in an image")
    hide_parser.add_argument("image_path", type=str, help="Path to the input image")
    hide_parser.add_argument("message", type=str, help="Message to hide in the image")
    hide_parser.add_argument("-p", "--password", type=str,  help="Password for encryption")
    hide_parser.add_argument("-m", "--method", type=str, default="auto", choices=["auto", "png_lsb", "png_lsb_matching", "jpeg_dct"], help="Method to use for hiding the message")
    
    extract_parser = subparsers.add_parser("extract", help="Extract a message from an image")
    extract_parser.add_argument("image_path", type=str, help="Path to the image with hidden message")
    extract_parser.add_argument("-p", "--password", type=str, help="Password used for encryption")
    extract_parser.add_argument("-m", "--method", type=str, default="auto", choices=["auto", "png_lsb", "jpeg_dct"], help="Method to use for extracting the message")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an image for steganographic content")
    analyze_parser.add_argument("image_path", type=str, help="Path to the image to analyze")
    analyze_parser.add_argument("-m", "--method", type=str, default="auto", choices=["auto", "png_lsb", "jpeg_dct"], help="Method to use for analysis")

    args = parser.parse_args()

    try:
        if args.command == "hide":
            res: dict = hide_message(args.image_path, args.message, args.password, args.method)
            print(f"Message hidden successfully.\nResult: {res}")
        elif args.command == "extract":
            message = extract_message(args.image_path, args.password, args.method)
            print(f"Extracted message:\n{message}")
        elif args.command == "analyze":
            res = analyze_image(args.image_path, args.method)
            print(f"--- Steganalysis Results ({args.method}) ---")
            print(f"Target File: {args.image_path}")
            print(f"Stego Probability (p-value): {res['stego_probability']:.4f}")
            print(f"Chi-Square Statistic: {res['chi2_stat']:.2f}")
            print(f"Degrees of Freedom: {res['dof']}")
            print(f"Stego Detected? -> {'YES (Anomalous payload detected)' if res['detected'] else 'NO (Natural statistics)'}")
        else:
            parser.print_help()
    except ValueError as e:
        if "MAC check failed" in str(e):
            print("Error: Incorrect password or corrupted data.", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()