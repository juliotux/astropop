from ..py_utils import check_iterable

__all__ = ['slices_from_string', 'string_from_slices']


def string_from_slices(slices, fits_convention=False):
    """Create a slice string from python slices.

    Fits convention not suported yet.
    """
    # TODO: implement fits convention
    if fits_convention:
        raise NotImplementedError('Fits convention is not supported yet.')

    if not check_iterable(slices):
        slices = [slices]

    for i in range(len(slices)):
        slices[i] = '{}:{}'.format(i.start or '', i.stop or '')
        slices[i] += '' if not i.step else ':{}'.format(i.step)

    return ','.join(slices)


def slices_from_string(section, fits_convention=False):
    """Generate a python slice from a string. Fits convention supported."""
    # Clean spaces and brakets
    sec = section.replace(' ', '')
    sec = sec.strip('[]')

    # Replace special cases of fits convention
    if fits_convention:
        sec = sec.replace('-*:', '::-')  # flip dimension with setp
        sec = sec.replace('-*', '::-1')  # flip dimension
        sec = sec.replace('*', ':')  # all

    slices = [slice(*(int(i) if i else None for i in s.split(':')))
              for s in sec.split(',')]

    # Fits convention require:
    # - subtract 1 from start index (start array from 1)
    # - do not subtract 1 from the end index (fits inclusive stop)
    # - flip slices order (faster first, like fortran)
    if fits_convention:
        slices = slices[::-1]
        for i in range(len(slices)):
            s = slices[i]
            start = s.start
            if start is not None:
                start = start - 1
            stop = s.stop
            step = s.step
            if start is not None and stop is not None:
                if stop < start:
                    # fits use positive step even with inverted dimensions
                    step = -1 if not step else step
                    step = -step if step > 0 else step
                    # prevent stop = -1
                    stop = None if stop == 1 else stop - 2
                if start < 0 or stop < 0:
                    raise ValueError('Minimum index of fits convention is '
                                     '1 and no negative stop allowed: '
                                     '{}:{}:{}'.format(start or '',
                                                       stop or '',
                                                       step or ''))
            slices[i] = slice(start, stop, step)

    return slices
