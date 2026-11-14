"""
TBRGS - Traffic-Based Route Guidance System for Boroondara (COS30019 Assignment 2B)

    python main.py                   open the GUI
    python main.py --cli -o 2000 -d 3002 -t "2006-10-26 08:00" [--model gru] [--method AS] [-k 5]
                                     same search on the command line
"""
import sys


def main() -> None:
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from src.Routing.route_finder import main as cli
        cli()
    else:
        from src.GUI.app import run
        run()


if __name__ == "__main__":
    main()
