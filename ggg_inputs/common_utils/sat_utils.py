import numpy as np


def time_weight_from_datetime(acos_datetime, prev_geos_datetime, next_geos_datetime):
    """
    Calculate the time weighting from datetime-like objects.

    :param acos_datetime: the date/time of the ACOS sounding
    :type acos_datetime: datetime-like

    :param prev_geos_datetime: the date/time of the GEOS FP file that comes before the ACOS sounding
    :type prev_geos_datetime: datetime-like

    :param next_geos_datetime: the date/time of the GEOS FP file that comes after the ACOS sounding
    :type next_geos_datetime: datetime-like

    :return: the weight, w, such that the profile for the ACOS time would be :math:`w*p1 + (1-w)*p2`, where p1 and p2
     are the profiles at the GEOS times before and after the sounding, respectively.
    :rtype: float
    """
    return time_weight(datetime2datenum(acos_datetime), datetime2datenum(prev_geos_datetime), datetime2datenum(next_geos_datetime))


def time_weight(acos_datenum, prev_geos_datenum, next_geos_datenum):
    """
    Calculate the time weighting from date numbers.

    In this context, a date number is any representation of a date time as a integer- or float-like value. Typically,
    this will be a value returned by :func:`datetime2datenum` but any similar numeric value will do, so long as all
    inputs are consistent.

    :param acos_datenum: the date/time of the ACOS sounding as a number
    :type acos_datenum: int or float

    :param prev_geos_datenum: the date/time of the GEOS FP file that comes before the ACOS sounding as a number
    :type prev_geos_datenum: int or float

    :param next_geos_datenum: the date/time of the GEOS FP file that comes after the ACOS sounding as a number
    :type next_geos_datenum: int or float

    :return: the weight, w, such that the profile for the ACOS time would be :math:`w*p1 + (1-w)*p2`, where p1 and p2
     are the profiles at the GEOS times before and after the sounding, respectively.
    :rtype: float
    """
    return (acos_datenum - prev_geos_datenum) / (next_geos_datenum - prev_geos_datenum)


def datetime2datenum(datetime_obj):
    """
    Convert a single :class:`datetime.datetime` object into a date number.

    Internally, this converts the datetime object into a :class:`numpy.datetime64` object with units of seconds, then
    converts that to a float type. Under numpy version 1.16.2, this results in a number that is seconds since 1 Jan
    1970; however, I have not seen any documentation from numpy guaranteeing that behavior. Therefore, any use of these
    date numbers should be careful to verify this behavior. The following assert block can be used to check this::

        assert numpy.isclose(datetime2datenum('1970-01-01'), 0.0) and numpy.isclose(datetime2datenum('1970-01-02'), 86400)

    :param datetime_obj: the datetime to convert. May be any type that :class:`numpy.datetime64` can intepret as a
     datetime.

    :return: the converted date number
    :rtype: :class:`numpy.float`
    """
    return np.datetime64(datetime_obj, 's').astype(np.float)