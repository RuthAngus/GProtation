__version__ = '0.1'

try:
    __GPROT_SETUP__
except NameError:
    __GPROT_SETUP__ = False

if not __GPROT_SETUP__:
    pass