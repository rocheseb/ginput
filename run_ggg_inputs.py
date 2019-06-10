#!/usr/bin/env python
import argparse

from ggg_inputs.priors import acos_interface as aci


def parse_args():
    parser = argparse.ArgumentParser(description='Call various pieces of ggg_inputs')
    subparsers = parser.add_subparsers(help='The following subcommands execute different parts of ggg_inputs')

    aci_parser = subparsers.add_parser('acos', help='Generate .h5 file for input into the OCO/GOSAT algorithm')
    aci.parse_args(aci_parser)

    return vars(parser.parse_args())


def main():
    args = parse_args()
    driver = args.pop('driver_fxn')
    driver(**args)


if __name__ == '__main__':
    main()
