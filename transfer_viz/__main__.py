from transfer_viz import teams, transfers


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "command",
        metavar="COMMAND",
        help=", ".join(sorted(["teams", "viz", "transfers"])),
    )

    args = parser.parse_args(sys.argv[1:2])

    # hack the program name for nested parsers
    sys.argv[0] += " " + args.command
    args.command = args.command.replace("_", "-")

    if args.command == "teams":
        teams.run(sys.argv[2:])
    if args.command == "transfers":
        transfers.run(sys.argv[2:])
    else:
        parser.print_help()
