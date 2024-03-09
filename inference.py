import argparse

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script that accepts a path.')

    # Add an argument
    parser.add_argument('--path', type=str, help='Path to the directory')

    # Parse the argument
    args = parser.parse_args()

    # Use the argument
    if args.path:
        print(f"The specified path is: {args.path}")
    else:
        print("No path was specified.")