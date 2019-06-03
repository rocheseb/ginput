from __future__ import print_function, division
import datetime as dt
import sys


_date_range_cl_help = 'The range of dates to get, in YYYYMMDD-YYYYMMDD format. The second date may be omitted, in ' \
                      'which case the end date will be one day after the first date. The end date is not included in ' \
                      'the range.'

_date_range_cl_help_extra = 'Either date may also be given as YYYYMMDD_hhmm, where the hhmm are hours and minutes.'


def eprint(msg, ecode=1):
    print(msg, file=sys.stderr)
    sys.exit(ecode)


def date_range_cl_help(allow_hours_minutes=True):
    if allow_hours_minutes:
        return _date_range_cl_help + ' ' + _date_range_cl_help_extra
    else:
        return _date_range_cl_help


def parse_date_range_no_hm(date_range_str):
    return parse_date_range(date_range_str, allow_hours_minutes=False)


def parse_date_range(date_range_str, allow_hours_minutes=True):
    def parse_date(date_str):
        if '_' in date_str:
            if not allow_hours_minutes:
                raise ValueError('Given datetime ({}) includes hours and minutes, but this is not allowed'
                                 .format(date_str))
            return dt.datetime.strptime(date_str, '%Y%m%d_%H%M')
        else:
            return dt.datetime.strptime(date_str, '%Y%m%d')

    parts = date_range_str.split('-')
    start_date = parse_date(parts[0])
    if len(parts) > 1:
        end_date = parse_date(parts[1])
    else:
        end_date = start_date + dt.timedelta(days=1)

    return start_date, end_date
