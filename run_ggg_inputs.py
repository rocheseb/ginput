#!/usr/bin/env python
import argparse

from ggg_inputs.priors import acos_interface as aci
from ggg_inputs.mod_maker import mod_maker
from ggg_inputs.download import get_GEOS5


def parse_args():
    parser = argparse.ArgumentParser(description='Call various pieces of ggg_inputs')
    subparsers = parser.add_subparsers(help='The following subcommands execute different parts of ggg_inputs')

    oco_parser = subparsers.add_parser('oco', help='Generate .h5 file for input into the OCO algorithm')
    aci.parse_args(oco_parser)

    # Albert Chang prefers to have a flag for whether reading OCO or GOSAT input files
    # acos_parser = subparsers.add_parser('acos', help='Generate .h5 file for input into the GOSAT algorithm')

    mm_parser = subparsers.add_parser('mod', help='Generate .mod (model) files for GGG')
    mod_maker.parse_args(mm_parser)

    get_g5_parser = subparsers.add_parser('getg5', help='Download GEOS5 FP or FP-IT data')
    get_GEOS5.parse_args(get_g5_parser)

    return vars(parser.parse_args())


def main():
    args = parse_args()
    driver = args.pop('driver_fxn')
    driver(**args)


if __name__ == '__main__':
    main()
