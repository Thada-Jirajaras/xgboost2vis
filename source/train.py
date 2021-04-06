import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--integers', type = int, default = -1)
    args = parser.parse_args()